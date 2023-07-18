import sys
import numpy as np

def check_solution(task, solution):
    if task.shape[0] != solution.shape[0]:
        print(task.shape[0], solution.shape[0])
    assert task.shape[0] == solution.shape[0]
    assert 4 * (task.shape[1] - 2) == solution.shape[1]
    answers = []
    for i in range(task.shape[0]):
        H = task[i, 0]
        W = task[i, 1]
        S = H * W
        s_all = 0
        for j in range (task.shape[1]-2):
            r = task[i, j + 2]
            x1, y1, x2, y2 = solution[i, 4*j], solution[i, 4*j+1], solution[i, 4*j+2], solution[i, 4*j+3]
            assert x1 <= x2
            assert y1 <= y2
            assert 0 <= x1
            assert x2 <= (W - 1)
            assert 0 <= y1
            if y2 > (H - 1):
                print(f"H: {H}, y2: {y2}")
            assert y2 <= (H - 1)            
            assert int(x1) == x1
            assert int(x2) == x2
            assert int(y1) == y1
            assert int(y2) == y2
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            rr = np.maximum(w, h) / np.minimum(w, h)
            if round(np.abs(rr - r), 4) > 0.1:
                print(f"rr: {rr}, r: {r}")
                print(solution[i])
                print("task: ", task[i])
            assert round(np.abs(rr - r), 4) <= 0.1
            s = w * h
            s_all += s
        s_all = np.sum(s_all)
        assert s_all <= S
        answers.append(S - s_all)
        
        for j in range (task.shape[1]-3):
            x1, y1, x2, y2 = solution[i, 4*j], solution[i, 4*j+1], solution[i, 4*j+2], solution[i, 4*j+3]
            for t in range (j+1, task.shape[1]-2):
                xx1, yy1, xx2, yy2 = solution[i, 4*t], solution[i, 4*t+1], solution[i, 4*t+2], solution[i, 4*t+3]
                dx = np.minimum(x2, xx2) - np.maximum(x1, xx1) + 1
                dy = np.minimum(y2, yy2) - np.maximum(y1, yy1) + 1
                
                intersect = (dx > 0) & (dy > 0)
                if not ~intersect:
                    print(task[i])
                    print(f"({xx1}, {yy1}); ({xx2}, {yy2})\n({x1}, {y1}); ({x2}, {y2})\ndx: {dx}, dy: {dy}, intersect: {intersect}")
                assert ~intersect
    return answers

task = np.genfromtxt(sys.argv[1], delimiter=",", skip_header=1) # [:10,:]
solution = np.genfromtxt(sys.argv[2], delimiter=",", skip_header=1)

a = check_solution(task, solution)

np.savetxt("solution.csv.square", a, delimiter=",")

mean = np.mean(a)

print(mean)

with open("solution.csv.metrics", "w") as mf:
  mf.write(str(mean))


