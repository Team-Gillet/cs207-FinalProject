import superautodiff as sad


def jacobian(variables, functions):
    """Returns the Jacobian matrix containing the first derivative of each input function with respect to each input variable"""
    derivatives = []
    
    # Case where functions is a list of AD objects
    if type(functions) is list:
        
        if len(functions) is 0:
            raise ValueError("Functions cannot be empty; input either an AutoDiffVector object or a list of AutoDiff objects")
        
        for function in functions:
            for variable in variables:
                derivatives.append(function.der[variable])
        
        return np.array(derivatives).reshape(len(functions), len(variables))
    
    # Case where functions is an ADV object
    else:
        if len(functions.objects) is 0:
            raise ValueError("Functions cannot be empty; input either an AutoDiffVector object or a list of AutoDiff objects")
        
        try:
            for key in list(functions.objects.keys()):
                for variable in variables:
                    derivatives.append((functions.objects[key].der[variable]))
            return np.array(derivatives).reshape(len(functions.objects), len(variables))
        
        # If neither case fits raise an error
        except:
            raise ValueError("Function inputs need to either be an AutoDiffVector object or an array of AutoDiff objects")
                    
                