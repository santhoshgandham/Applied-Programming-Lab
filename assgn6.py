import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(42)

def readfile(filename):
    data = open(filename, 'r')
    data.readline()
    text = data.readlines()

    cities = []

    for line in text:
        parts = line.split()
        x = float(parts[0])
        y = float(parts[1])
        cities.append((x, y))  # a list of tuples

    return cities

cities = readfile("tsp40.txt")

def tsp(cities):
    # Calculate the distance from the first city to all other cities and sort them
    dict1 = {}
    for i in range(len(cities)):
        distance = (cities[0][0] - cities[i][0]) ** 2 + (cities[0][1] - cities[i][1]) ** 2
        dict1[distance] = i
    cityorder = sorted(dict1.items())
    cityorder = [b for (a, b) in cityorder]
    return cityorder

cityorder = tsp(cities)  # Get the initial city order

def distance(cityorder, cities):
    
    # Create separate arrays for x and y coordinates of the cities
    x_cities, y_cities = zip(*cities)

    def distcost(x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def totcost(x_cities, y_cities, cityorder):
        tot = 0
        for i in range(1, len(x_cities)):
            tot += distcost(x_cities[cityorder[i-1]], y_cities[cityorder[i-1]], x_cities[cityorder[i]], y_cities[cityorder[i]])
        tot += distcost(x_cities[cityorder[-1]], y_cities[cityorder[-1]], x_cities[cityorder[0]], y_cities[cityorder[0]])
        return tot
    
    tot=totcost(x_cities, y_cities, cityorder)
    print("Total distance which is not optimised:",tot)


    def simulated_annealing(x_cities, y_cities, cityorder, initial_temperature, cooling_rate, iterations):
        current_order = cityorder
        current_distance = totcost(x_cities, y_cities, current_order)
        best_order = current_order
        best_distance = current_distance
        temperature = initial_temperature

        for i in range(iterations):
            # Randomly select two distinct cities to swap
            rand_indices = np.random.choice(len(current_order), 2, replace=False)
            new_order = current_order.copy()
            new_order[rand_indices[0]], new_order[rand_indices[1]] = new_order[rand_indices[1]], new_order[rand_indices[0]]

            new_distance = totcost(x_cities, y_cities, new_order)

            if new_distance < current_distance or np.random.rand() < np.exp((current_distance - new_distance) / temperature):
                current_order = new_order
                current_distance = new_distance

                if current_distance < best_distance:
                    best_order = current_order
                    best_distance = current_distance

            temperature *= cooling_rate

        return best_order, best_distance

    best_order, best_distance = simulated_annealing(x_cities, y_cities, cityorder, initial_temperature=100000, cooling_rate=0.9995, iterations=100000)

    xplot = [x_cities[i] for i in best_order]
    yplot = [y_cities[i] for i in best_order]
    xplot.append(xplot[0])
    yplot.append(yplot[0])

    print(f"Total distance on optimisation       : {best_distance}")

    plt.clf()
    plt.plot(xplot, yplot, 'o-')
    plt.show()
    percent_improvement = ((tot-best_distance)/tot)*100
    return percent_improvement 


print(f"Percentage improvement in the path   : {distance(cityorder, cities)}%")


