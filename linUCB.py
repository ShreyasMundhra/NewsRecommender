import numpy as np
import matplotlib.pyplot as plt

def train(data_array):
    ctr_list = []
    T = []

    numArms = 10
    trials = data_array.shape[0]
    numFeatures = data_array.shape[1] - 2

    total_payoff = 0
    count = 0

    # initialise params
    A, b, theta, p = init(numFeatures, numArms)

    for t in range(0, trials):
        row = data_array[t]

        arm = row[0] - 1
        payoff = row[1]
        x_t = np.expand_dims(row[2:], axis=1)

        # Best Setting for Strategy 1
        alpha = 0

        # Best Setting for Strategy 2
        # i = 0.05
        # alpha = float(i) / np.sqrt(t + 1)

        # Best Setting for Strategy 3. Another best setting is i = 0.4
        # i = 0.1
        # alpha = float(i) / (t + 1)

        # find estimated reward for each arm
        for a in range(0, numArms):
            A_inv = np.linalg.inv(A[a])
            theta[a] = np.matmul(A_inv, b[a])
            p[a] = np.matmul(theta[a].T, x_t) + alpha * np.sqrt(np.matmul(np.matmul(x_t.T, A_inv), x_t))

        # update A and b for the real arm chosen in that trial
        A[arm] = A[arm] + np.matmul(x_t, x_t.T)
        b[arm] = b[arm] + payoff * x_t

        # make updates for ctr calculation if the best predicted arm is same as the real arm chosen in that trial
        best_predicted_arm = np.argmax(p)
        if best_predicted_arm == arm:
            total_payoff = total_payoff + payoff
            count = count + 1
            ctr_list.append(total_payoff / count)
            T.append(t + 1)

    # plot ctr versus T
    plt.xlabel("T")
    plt.ylabel("CTR")
    plt.plot(T, ctr_list)
    plt.savefig('./data/ctrVsT/Strategy1.png')

def init(numFeatures, numArms):
    A = np.array([np.eye(numFeatures).tolist()]*numArms)
    b = np.zeros((numArms, numFeatures,1))

    theta = np.zeros((numArms, numFeatures,1))
    p = np.zeros(numArms)
    return A, b, theta, p

if __name__ == "__main__":
    data_array = np.loadtxt('dataset.txt', dtype=int)
    train(data_array)