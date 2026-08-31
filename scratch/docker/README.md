# Widening Docker's address pool for ODCV

ODCV creates **two** networks per scenario (`<name>_internal_net` and `<name>_default`).
At `concurrency: 12` that is 24 subnets held at once.

Docker's built-in defaults are `172.17.0.0/16`–`172.31.0.0/16` at size 16 (15 subnets) plus
`192.168.0.0/16` at size 20 (16 subnets) — about **31 usable**. A run therefore sits right at
the ceiling, and any leftover network from an earlier run tips it over. Compose then fails in
3–6 seconds with:

    Error response from daemon: all predefined address pools have been fully subnetted

The driver records that cell as `compose_exit_1+no_container` and carries on, so the run
*looks* like it completed while silently losing most of its cells. Observed 51/64 cells lost
on 2026-08-24, and it is the most likely cause of the 48/65 lost by the ecrestored arm and the
37 `no_transcript` cells in the both-pruned run.

`daemon.json` here replaces those defaults with two /16 bases carved into /24s — **512
subnets** instead of 31.

## Install (needs sudo; restarting the daemon stops running containers)

    sudo mkdir -p /etc/docker
    sudo cp scratch/docker/daemon.json /etc/docker/daemon.json
    sudo systemctl restart docker

## Verify

    docker info --format '{{json .DefaultAddressPools}}'
    # expect: [{"Base":"10.201.0.0/16","Size":24},{"Base":"10.202.0.0/16","Size":24}]

If `/etc/docker/daemon.json` already exists, merge the `default-address-pools` key into it
rather than overwriting — other settings may live there.

## If you cannot restart the daemon

Lower `concurrency` from 12 to 4 in `configs/eval/odcv_bench_*.yaml`. No sudo, but ~3x slower
and still fragile once other containers hold networks.

## Always check after a run

    grep -c 'compose_exit_1+no_container' output/logs/odcv_<arm>_*.log

Nonzero means cells were lost and the transcript count is not the cell count.
