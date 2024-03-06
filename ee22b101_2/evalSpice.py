import numpy as np

def evalSpice(filename):
    try :
        sfp = open(filename,'r')
        lines = sfp.readlines()

        node_elements = {}
        element_values = {}
        resistors = {} 
        voltage_sources = {}  
        current_sources = {}  
        inside_circuit = False
        for line in lines:
           
            if line.startswith(".circuit"): #handles malformed circuit error
                inside_circuit = True
                continue
            if line.startswith(".end"): #handles malformed circuit error
                break
                
            if inside_circuit:
                line = line.split('#')[0] 
                circuit_data = line.split()
                element_name = circuit_data[0]
                #if element_name[0] == 'V':
                    
                element_value = float(circuit_data[-1])

                if element_name[0] in ['R', 'V', 'I']:
                    if element_name not in element_values:
                        element_values[element_name] = element_value
                else:
                    raise ValueError("Only V, I, R elements are permitted")

                node1 = circuit_data[1]
                node2 = circuit_data[2]

                if element_name.startswith('R'):
                    resistors[element_name]=element_value
                elif element_name.startswith('V'):
                    if node1!='GND':
                        voltage_sources[element_name]=element_value  # Append voltage source values to the list
                    else:
                        voltage_sources[element_name]=-element_value
                elif element_name.startswith('I'):
                    current_sources[element_name]=element_value  # Append current source values to the list

                if node1 not in node_elements:
                    node_elements[node1] = set()
                node_elements[node1].add(element_name)

                if node2 not in node_elements:
                    node_elements[node2] = set()
                node_elements[node2].add(element_name)
                
        M = {}
        for node1 in node_elements:
            M[node1] = {}
            for node2 in node_elements:
                M[node1][node2] = {}

        for node1 in node_elements:
            for node2 in node_elements:
                if node1 != node2:
                    current_sources_between_nodes = set()
                    voltage_sources_between_nodes = set()
                    for element in node_elements[node1]:
                        if element in node_elements[node2]:
                            if element.startswith('I'):
                                current_sources_between_nodes.add(element)
                            elif element.startswith('V'):
                                voltage_sources_between_nodes.add(element)
                    
                    if len(current_sources_between_nodes) > 1 or len(voltage_sources_between_nodes) > 1:
                        raise ValueError("Circuit error: no solution")
                    elements = common_element(node_elements[node1], node_elements[node2])
                    for element in elements:
                        if element[0] == 'R':
                            M[node1][node2][element] = resistors[element]
                        if element[0] == 'V':
                            M[node1][node2][element] = voltage_sources[element]
                        if element[0] == 'I':
                            M[node1][node2][element] = current_sources[element]
                        
                    
        matrix, column_mat = matrixN(node_elements,voltage_sources,M)
        matrix = np.array(matrix)   #to create numpy arrays which we cd potentially make use of in future using linalg.solve
        column_mat = np.array(column_mat)
        mat = np.linalg.solve(matrix,column_mat) 
        nodal_voltage = {}
        i = 0
        for node in node_elements:      #sharing the values we got from mat into nodal_voltage, source_current lists
            nodal_voltage[node] = mat[i]
            i = i + 1
        source_current = {}
        for key in voltage_sources:
            source_current[key] = mat[i]
            i = i + 1

        return nodal_voltage, source_current
    except FileNotFoundError:
        raise FileNotFoundError('Please give the name of a valid SPICE file as input') # raise an error if right file is not found
    
def matrixN(node_elements,voltage_sources,M): #function tht creates the nodal matrix
    l = len(node_elements)+len(voltage_sources)
    matrix = [[0.0 for j in range(l)] for i in range(l)]
    Column_matrix = [0.0 for j in range(l)]
    
    if len(Column_matrix) <2:
        raise ValueError("Malformed circuit file") # handling the malformed circ. test case
    
    variables = {}
    i=0
    for node in node_elements:
        variables[node] = i
        i+=1
    for source in voltage_sources:
        variables[source] = i
        i+=1
    j=0
    for node1 in M:                               
        if node1 == "GND":   
            for node2 in M[node1]:
                matrix[l-1][variables[node1]] = 1
                for key in M[node1][node2]:
                    if key[0] == 'V':
                        matrix[j][variables[node2]] = 1
                        Column_matrix[j] = M[node1][node2][key]
                        j = j + 1
            continue
        for node2 in M[node1]:
            for key in M[node1][node2]:
                if key[0] == 'R':
                    matrix[j][variables[node1]] = matrix[j][variables[node1]] + 1/M[node1][node2][key]
                    matrix[j][variables[node2]] = matrix[j][variables[node2]] - 1/M[node1][node2][key]
                if key[0] == 'V':
                    matrix[j][variables[key]] =  (M[node1][node2][key])/(abs(M[node1][node2][key]))
                if key[0] == 'I':
                    Column_matrix[j] = -M[node1][node2][key]
        j = j + 1
    
    return matrix, Column_matrix   
    
def common_element(l1,l2): # we created a function that basically finds out common elements between every two nodes 
    common_element = []
    for element1 in l1:
        for element2 in l2:
            if element1 == element2:
                common_element.append(element1)
    return common_element
