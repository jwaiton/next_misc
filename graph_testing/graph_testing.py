import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def main():
    g = nx.petersen_graph()
    subax1 = plt.subplot(121)
    nx.draw(g, with_labels = True, font_weight = 'bold')
    print(g)
    print(g.nodes.hits())
    plt.show()

main()
