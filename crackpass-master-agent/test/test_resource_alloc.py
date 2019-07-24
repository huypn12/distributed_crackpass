


def create_node_list():
    node_list = []
    node = {}
    node['node_id']   = 0
    node['amqp_url']  = "fake_link"
    node['avail_cpu'] = 1
    node['avail_gpu'] = 8
    for i in range(1,8):
        node['node_id'] = i
        node_list.append(node)


def create_task_list():
    task_list = []
    task = {}
    task['n_cpu'] = 4
    task['n_gpu'] = 4
    for i in range(1,8):
        task['task_id'] = i
        task_list.append(task)


def alloc_node_resources(node_list, candidate_task):
    # Calculate allocation
    needed_cpu = candidate_task['n_cpu']
    needed_gpu = candidate_task['n_gpu']
    worker_list = []
    node_idx = i
    cpu_prior_node_list =
    while needed_cpu > 0 or needed_gpu > 0:
        # No need to check idx condition here
        # -- checked at preconditional test
        node = cpu_prior_node_list[node_idx]
        if node['avail_cpu'] == 0 and node['avail_gpu'] == 0:
            node_idx += 1
            continue
        # Calculating CPU allocation
        node_alloc_cpu = 0
        if needed_cpu > 0:
            node_alloc_cpu = node['avail_cpu']
            if needed_cpu <= node['avail_cpu']:
                node_alloc_cpu = needed_cpu
                node['avail_cpu'] -= needed_cpu
                needed_cpu = 0
            else:
                needed_cpu -= node['avail_cpu']
                node_alloc_cpu = node['avail_cpu']
                node['avail_cpu'] = 0
        # Calculating GPU allocation
        node_alloc_gpu = 0
        if needed_gpu > 0:
            node_alloc_gpu = node['avail_gpu']
            if needed_gpu <= node['avail_gpu']:
                node_alloc_gpu = needed_gpu
                node['avail_gpu'] -= needed_gpu
                needed_gpu = 0
            else:
                needed_gpu -= node['avail_gpu']
                node_alloc_gpu = node['avail_gpu']
                node['avail_gpu'] = 0
        # Append calculated allocation to worker list
        worker = {}
        #@huypn: 2015-10-14 00:00 repair mistyped
        # worker['node_id'] = node_idx
        worker['node_id'] = node['node_id']
        worker['amqp_url'] = node['amqp_url']
        worker['n_cpu'] = node_alloc_cpu
        worker['n_gpu'] = node_alloc_gpu
        worker['avail_cpu'] = node['avail_cpu']
        worker['avail_gpu'] = node['avail_gpu']
        worker_list.append(worker)
        # Increase loop step
        node_idx += 1

