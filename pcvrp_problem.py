from qubots.base_problem import BaseProblem
import math, random, sys, os

PENALTY = 10**9

def read_elem(filename):

    # Resolve relative path with respect to this module’s directory.
    if not os.path.isabs(filename):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(base_dir, filename)

    with open(filename) as f:
        return [str(elem) for elem in f.read().split()]

def compute_distance_matrix(customers_x, customers_y):
    nb_customers = len(customers_x)
    matrix = [[0 for _ in range(nb_customers)] for _ in range(nb_customers)]
    for i in range(nb_customers):
        matrix[i][i] = 0
        for j in range(i+1, nb_customers):
            d = compute_dist(customers_x[i], customers_x[j], customers_y[i], customers_y[j])
            matrix[i][j] = d
            matrix[j][i] = d
    return matrix

def compute_distance_depots(depot_x, depot_y, customers_x, customers_y):
    nb_customers = len(customers_x)
    dist = [0]*nb_customers
    for i in range(nb_customers):
        d = compute_dist(depot_x, customers_x[i], depot_y, customers_y[i])
        dist[i] = d
    return dist

def compute_dist(xi, xj, yi, yj):
    exact = math.sqrt((xi - xj)**2 + (yi - yj)**2)
    return int(math.floor(exact + 0.5))

def read_input_pcvrp(filename):
    file_it = iter(read_elem(filename))
    nb_trucks = int(next(file_it))
    truck_capacity = int(next(file_it))
    demands_to_satisfy = int(next(file_it))
    
    n = 0
    customers_x = []
    customers_y = []
    depot_x = 0
    depot_y = 0
    demands = []
    prizes = []
    
    token = next(file_it, None)
    while token is not None:
        node_id = int(token)
        if node_id != n:
            print("Unexpected index")
            sys.exit(1)
        if n == 0:
            depot_x = int(next(file_it))
            depot_y = int(next(file_it))
        else:
            customers_x.append(int(next(file_it)))
            customers_y.append(int(next(file_it)))
            demands.append(int(next(file_it)))
            prizes.append(int(next(file_it)))
        token = next(file_it, None)
        n += 1
    
    # Compute distance matrix among customers.
    distance_matrix = compute_distance_matrix(customers_x, customers_y)
    # Compute distances from depot to each customer.
    distance_depots = compute_distance_depots(depot_x, depot_y, customers_x, customers_y)
    nb_customers = n - 1
    return nb_customers, nb_trucks, truck_capacity, distance_matrix, distance_depots, demands, demands_to_satisfy, prizes

class PCVRPProblem(BaseProblem):
    """
    Prize-Collecting Vehicle Routing Problem (PCVRP) for Qubots.
    
    A fleet of vehicles with uniform capacity must serve a set of customers starting and ending at a common depot.
    Each customer has a demand and a prize. Each customer is served by at most one vehicle.
    In addition to serving customers, the total quantity delivered must be at least a predetermined amount.
    The objectives, in lexicographic order, are:
      1. Minimize the number of trucks used.
      2. Maximize the total prize collected.
      3. Minimize the total distance traveled.
      
    These are combined into one scalar objective:
    
         Objective = (nb_trucks_used * 1e9) - (total_prize * 1e6) + total_distance
    
    A candidate solution is a dictionary with key "routes" mapping to a list of length nb_trucks.
    Each route is a list of customer indices (0-indexed) representing the order in which that truck visits customers.
    """
    
    def __init__(self, instance_file: str, **kwargs):
        (self.nb_customers, self.nb_trucks, self.truck_capacity, 
         self.dist_matrix, self.dist_depot, self.demands, 
         self.demands_to_satisfy, self.prizes) = read_input_pcvrp(instance_file)
    
    def evaluate_solution(self, solution) -> int:
        # Expect solution to be a dict with key "routes" (list of length nb_trucks)
        if not isinstance(solution, dict) or "routes" not in solution:
            return PENALTY
        routes = solution["routes"]
        if not isinstance(routes, list) or len(routes) != self.nb_trucks:
            return PENALTY
        
        # Check that each customer (indices 1...nb_customers) is visited exactly once.
        # Note: In our instance, customers are indexed 1...nb_customers (since depot is node 0)
        # However, our read_input_pcvrp treats depot as index 0 and customers as indices 1..n-1.
        # Here, we assume customer indices are 1...nb_customers.
        assigned = []
        for route in routes:
            if not isinstance(route, list):
                return PENALTY
            assigned.extend(route)
        if sorted(assigned) != list(range(1, self.nb_customers + 1)):
            return PENALTY
        
        total_route_distance = 0
        total_route_prize = 0
        total_quantity = 0
        nb_trucks_used = 0
        # For each truck route:
        for route in routes:
            if len(route) == 0:
                continue
            nb_trucks_used += 1
            # Compute total demand on the route.
            route_quantity = sum(self.demands[i-1] for i in route)  # demands list is 0-indexed for customer1..n
            if route_quantity > self.truck_capacity:
                return PENALTY
            total_quantity += route_quantity
            # Compute route distance: from depot (index 0) to first customer, then between customers, then back to depot.
            # In our distance matrices, the depot is at index 0 and customers are at indices 1...nb_customers.
            d_route = self.dist_depot[route[0]-1]  # depot to first customer
            for i in range(1, len(route)):
                d_route += self.dist_matrix[route[i-1]-1][route[i]-1]
            d_route += self.dist_depot[route[-1]-1]
            if d_route > 0 and d_route < 0:  # never happens; just placeholder for potential constraint on max distance.
                return PENALTY
            total_route_distance += d_route
            # Compute route prize.
            route_prize = sum(self.prizes[i-1] for i in route)
            total_route_prize += route_prize

        if total_quantity < self.demands_to_satisfy:
            return PENALTY

        # Lexicographic objective: minimize trucks used, maximize prize, minimize distance.
        # We combine using large weights:
        objective = nb_trucks_used * 10**9 - total_route_prize * 10**6 + total_route_distance
        return objective

    def random_solution(self):
        # Generate a random partition of customers among the trucks.
        routes = [[] for _ in range(self.nb_trucks)]
        # Customers are numbered from 1 to nb_customers.
        for cust in range(1, self.nb_customers + 1):
            r = random.randrange(self.nb_trucks)
            routes[r].append(cust)
        for route in routes:
            random.shuffle(route)
        return {"routes": routes}