import pandas as pd
import numpy as np
import matplotlib.pyplot as plt








"""# Q1 write a program take 1d integer array that convert into numpy and panda  
print("Q1 : ")
arr = np.array((input("Enter 1d array elements :[[a,b],[b,c]] :")))
print(arr)

# Q2 write a program take 2d integer array convert to 2d array using numpy 


print("Q2 : ")
arr = np.array(eval(input("Enter 2d array elements  :[[a,b],[b,c]] :")))

print("Converting 2d input array to 1d array")

arr_1d = arr.flatten()
print(arr_1d)
"""

# Q3 take 5 diff coord value and find out the total error in linear regression

#dataframe to store coordinates
df= pd.DataFrame(columns=["X","Y"],)

#inputs
coords = eval(input("Enter 5 coordinates [(x1,y1),(x2,y2)] :"))


for x, y in coords:
    df.loc[len(df)] = [x, y]

print("\nCoordinates DataFrame:")
print(df)


#list of x and y coords 
x = df["X"].values
y = df["Y"].values


y_pred = 2*x + 5

total_error = np.sum(abs((y - y_pred)))

for Y in y_pred:
    p1=[(x,Y)]
for Y in y:
    p2=[(x,Y)]



print("\nTaken Line: y = {:.3f}x + {:.3f}".format(2,5))
print("Total absolute Error:", total_error)

plt.scatter(x, y)          
plt.plot(x, y_pred) 

for i in range(5):
    x_vals = [p1[i][0],p2[i][0]]
    y_vals = [p1[i][1],p2[i][1]]
    plt.plot(x_vals,y_vals)

plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Linear Regression Fit")
plt.show()




