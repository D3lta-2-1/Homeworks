import matplotlib.pyplot as plt
import sys

def syracuse(n: int) -> int:
    if n % 2 == 0:
        return n // 2
    else:
        return 3 * n + 1
    
def make(list: list, n: int) -> list:
    list.append(n)
    if n == 1:
        return list
    else:
        return make(list, syracuse(n))

def syracuse_list(n: int) -> list:
    return make([], n)

def main(start: int):
    y = syracuse_list(start)
    x = list(range(1, len(y) + 1))
    plt.plot(x, y)
    plt.show()



if __name__ == "__main__":
    sys.setrecursionlimit(10**6)
    main(95_647_806_479_275_528_135_733_781_266_203_904_794_419_563_064_407)