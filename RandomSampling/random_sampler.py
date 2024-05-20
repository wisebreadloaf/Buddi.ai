import numpy as np


def drawSamples(pmf: dict[str, float], n: int = 5) -> list[str]:
    cmf = []
    key_list = list(pmf.keys())
    for key in key_list:
        index = key_list.index(key)
        if index == 0:
            cmf.append(pmf[key])
        else:
            cmf.append(pmf[key] + cmf[index - 1])
    samples = []

    for _ in range(n):
        random_var = np.random.uniform(0, 1)
        flag = False
        for i in range(len(cmf)):
            if random_var < cmf[i] and not flag:
                samples.append(key_list[i])
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
print(drawSamples(pmf, 16))
