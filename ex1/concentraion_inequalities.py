import numpy
from matplotlib.pyplot import *


def estimate_xm(data):
    for i in range(0, 5):
        m_value, xm = [], []
        for m in range(1000):
            m_value.append(m)
            xm.append(sum(data[i][:m + 1]) / (m + 1))
        plot(m_value, xm)
    xlabel('mean of m tosses', fontweight="bold")
    ylabel('probability', fontweight="bold")
    legend(('row 0', 'row 1', 'row 2', 'row 3', 'row 4'), loc='upper right')
    show()


def bound(epsilon):
    variance = 0.25 * 0.75
    percentage = np.abs((np.cumsum(data, 1) / np.array([i for i in range(1, 1001)], dtype=int)) - 0.25)
    for e in epsilon:
        hoeffding, chebyshev, m_value = [], [], []
        for m in range(1, 1001):
            m_value.append(m)
            hoeffding.append(min(1.0, 2 * numpy.exp(-2 * m * (e ** 2))))
            chebyshev.append(min(1.0, variance / (m * (e ** 2))))
        plot(m_value, hoeffding, m_value, chebyshev, m_value, np.sum(percentage >= e, 0) / 100000)
        xlabel('m value', fontweight="bold")
        ylabel('probability', fontweight="bold")
        legend(('hoeffding', 'chebyshev', 'percentage of sequences'), loc='upper right')
        show()


if __name__ == '__main__':
    data = numpy.random.binomial(1, 0.25, (100000, 1000))
    estimate_xm(data)
    epsilon = [0.5, 0.25, 0.1, 0.01, 0.001]
    bound(epsilon)
