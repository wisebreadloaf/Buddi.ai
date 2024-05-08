import numpy as np


def sampler(pmf, num_sample=5):
    cmf = {}
    keys_list = list(pmf.keys())
    for key in pmf.keys():
        index = keys_list.index(key)
        if index == 0:
            cmf[key] = pmf[key]
        else:
            cmf[key] = pmf[key] + cmf[keys_list[index - 1]]
    samples = []
    for _ in range(num_sample):
        random_var = np.random.uniform(0, 1)
        flag = False
        for key in cmf.keys():
            if random_var < cmf[key] and not flag:
                samples.append(key)
                flag = True
    return samples


pmf = {
    "ABCD": 0.03703704,
    "EFGH": 0.07407407,
    "IJKL": 0.14814815,
    "MNOP": 0.22222222,
    "QRST": 0.40740741,
    "UVWXYZ": 0.11111111,
}
print(sampler(pmf, 6))
