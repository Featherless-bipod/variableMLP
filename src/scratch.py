for N in neurons:
    acc, dur = t.train_and_evaluate(H=2, N=N, train_x=train_x, train_y=train_y,
                                  test_x=test_x, test_y=test_y,
                                  pixels=pixels, results=10, iters=500, alpha=0.01)
    accs.append(acc)
    times.append(dur)

plt.plot(neurons, accs, marker='o')
plt.xlabel("Neurons per hidden layer")
plt.ylabel("Test Accuracy")
plt.title("Accuracy vs Hidden Layer Width")
plt.show()