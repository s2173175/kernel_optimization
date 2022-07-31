
from subprocess import Popen


def run_com():

    base_name = 'walking_cmd_com'

    experiments = [
        # 'walking_cmd_2',
        # 'walking_cmd_25',
        # 'walking_cmd_3',
        # 'walking_cmd_2_com',
        'walking_cmd_25_com',
        # 'walking_cmd_3_com',
        # 'walking_cmd_2_l_a',
    ]

    input_dims = [
        # 7,
        # 7,
        # 7,
        # 10,
        10,
        # 10,
        # 13
    ]

    commands = []

    for i, exp in enumerate(experiments):
        commands.append(f'python train_auto.py \
            -name {exp} \
            -input_dims {input_dims[i]}')

    workers = 1
    for i in range(5):
        c = [commands[i+w] for w in range(workers)]
        processes = [Popen(cmd, shell=True) for cmd in c]
        for p in processes: p.wait()

    return



# def run_com_push():
#     base_name = 'walking_cmd_com_push'
#     experiments = ['', '_l', '_l_a']
#     input_dims = [10,13,16]

#     commands = []

#     for i, exp in enumerate(experiments):
#         commands.append(f'python train_auto.py \
#             -name {base_name+exp} \
#             -input_dims {input_dims[i]}')

#     workers = 1
#     for i in range(3):
#         c = [commands[i+w] for w in range(workers)]
#         processes = [Popen(cmd, shell=True) for cmd in c]
#         for p in processes: p.wait()

#     return


run_com()
# run_com_push()