import torch
import matplotlib.pyplot as plt

@torch.no_grad()
def plot_flow_results(
    net,
    src_sampler,
    tgt_sampler,
    num_samples=1000,
    plot_trajs: bool = False,
    traj_count: int = 50,
    K: int = 50,
    figsize=(6,6),
    xlim=(-4,4),
    ylim=(-4,4),
):
    """
    Integrate ẋ = v(x,t) from t=0 to t=1 in K steps and plot:
      • Source points (blue)
      • Target points (orange)
      • Integrated flow endpoints (green)
      • Optional full trajectories (grey)
    """
    device = next(net.parameters()).device

    # 1) sample raw points
    x0 = src_sampler.generate(num_samples).to(device)     # (N,2)
    xtgt = tgt_sampler.generate(num_samples).to(device)   # for background
    dt = 1.0 / K

    # 2) integrate flow for ALL points if plotting endpoints only,
    #    or for a subset if plotting trajectories.
    if plot_trajs:
        # integrate only traj_count histories
        idx = torch.randperm(num_samples)[:traj_count]
        x_hist = torch.zeros(traj_count, K+1, x0.shape[1], device=device)
        x_curr = x0[idx].clone()
        x_hist[:,0] = x_curr
        for k in range(1, K+1):
            t = torch.full((traj_count,1), (k-1)*dt, device=device)
            v = net(x_curr, t)
            x_curr = x_curr + v * dt
            x_hist[:,k] = x_curr
        # integrate endpoints for all
        x_end = x0.clone()
        for k in range(K):
            t = torch.full((num_samples,1), k*dt, device=device)
            x_end = x_end + net(x_end, t) * dt
    else:
        # only need endpoints
        x_end = x0.clone()
        for k in range(K):
            t = torch.full((num_samples,1), k*dt, device=device)
            x_end = x_end + net(x_end, t) * dt

    # move to CPU/NumPy for plotting
    x0_np   = x0.cpu().numpy()
    xt_np   = xtgt.cpu().numpy()
    xend_np = x_end.cpu().numpy()

    if plot_trajs:
        xh_np = x_hist.cpu().numpy()  # (traj_count, K+1, 2)

    # 3) plot
    plt.figure(figsize=figsize)
    ax = plt.gca()

    ax.scatter(x0_np[:,0],    x0_np[:,1],    s=8, alpha=0.6, label='Source')
    ax.scatter(xt_np[:,0],    xt_np[:,1],    s=8, alpha=0.6, label='Target')
    ax.scatter(xend_np[:,0],  xend_np[:,1],  s=8, alpha=0.6, label='Flow Endpoints')

    if plot_trajs:
        for i in range(traj_count):
            ax.plot(
                xh_np[i,:,0],
                xh_np[i,:,1],
                linewidth=1.0,
                alpha=0.5,
                color='grey'
            )

    # 4) styling
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title("Flow Integration Results", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_aspect('equal')
    ax.legend(fontsize=12, loc='upper left')
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def plot_meanflow_results(
    net,
    src_sampler,
    tgt_sampler,
    num_samples=1000,
    plot_trajs: bool = False,
    traj_count: int = 50,
    K: int = 10,
    figsize=(6,6),
    xlim=(-4,4),
    ylim=(-4,4),
):
    """
    Visualize Mean-Flow integration over K subintervals:
      • Source points (blue)
      • Target points (orange)
      • Mean-flow endpoints (green)
      • Optional full trajectories (grey)

    net: MeanFlowNet, expects (x, r, t) → uθ
    """
    device = next(net.parameters()).device

    # 1) sample data
    x0   = src_sampler.generate(num_samples).to(device)   # (N,2)
    xtgt = tgt_sampler.generate(num_samples).to(device)   # for background

    # 2) prepare r_k, t_k for k=0..K-1
    #    splits at [0, 1/K, 2/K, ..., 1]
    edges = torch.linspace(0., 1., K+1, device=device)    # (K+1,)
    r_list = edges[:-1].unsqueeze(1)                      # (K,1)
    t_list = edges[1:].unsqueeze(1)                       # (K,1)
    del edges

    # 3) integrate mean-flow
    if plot_trajs:
        # track histories for traj_count points
        idx = torch.randperm(num_samples)[:traj_count]
        hist = torch.zeros(traj_count, K+1, x0.size(1), device=device)
        x_curr = x0[idx].clone()
        hist[:,0] = x_curr

        # we’ll still compute endpoints for all N below
        x_end = x0.clone()
    else:
        x_end = x0.clone()

    # loop over segments
    for k in range(K):
        r_k = r_list[k].expand_as(x_end[:, :1])            # shape (N,1)
        t_k = t_list[k].expand_as(r_k)                     # shape (N,1)
        delta = (t_k - r_k)                                # scalar step

        # update all endpoints
        u_all = net(x_end, r_k, t_k)                       # (N,2)
        x_end = x_end + delta * u_all

        if plot_trajs:
            # update histories
            r_k_t = r_list[k].expand(traj_count,1)
            t_k_t = t_list[k].expand(traj_count,1)
            u_sub = net(x_curr, r_k_t, t_k_t)              # (traj_count,2)
            x_curr = x_curr + (t_k_t - r_k_t) * u_sub
            hist[:, k+1] = x_curr

    # 4) move to CPU/NumPy
    x0_np   = x0.cpu().numpy()
    xt_np   = xtgt.cpu().numpy()
    xend_np = x_end.cpu().numpy()

    if plot_trajs:
        hist_np = hist.cpu().numpy()  # (traj_count, K+1, 2)

    # 5) plot
    plt.figure(figsize=figsize)
    ax = plt.gca()

    ax.scatter(x0_np[:,0],    x0_np[:,1],    s=8, alpha=0.6, label='Source')
    ax.scatter(xt_np[:,0],    xt_np[:,1],    s=8, alpha=0.6, label='Target')
    ax.scatter(xend_np[:,0],  xend_np[:,1],  s=8, alpha=0.6, label='Mean‑Flow Endpoints')

    if plot_trajs:
        for i in range(traj_count):
            ax.plot(
                hist_np[i,:,0],
                hist_np[i,:,1],
                linewidth=1.0,
                alpha=0.5,
                color='grey'
            )

    # 6) styling
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(f"Mean‑Flow Integration (K={K})", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_aspect('equal')
    ax.legend(fontsize=12, loc='upper left')
    plt.tight_layout()
    plt.show()