# Finite_Element_Analysis

Description: 

This FEA package contains functions to carry out a 2D fininte element analysis on a mesh consisting of nodes and elements. The nodes and elements are assembled manually or through included mesh generators (ex. gen_bar()) and then the matrix problem is solved through the assemble() and solve_system() functions. The package also contains functions for setting restraints in the x and y direction as well as assigning point loads. 


The package currently contains implementation for triangular 3 node elements (T3) and quadtrilateral 4 node elements (Q4). 


The solution can be solved as a linear elastic problem or through the hybrid formulation assuming constant pressure per element. 


NumPy is the primary package used to implement the matrix calculation/storage of the system. 
