''' Q.2) For a given set of training data examples stored in a .CCSV file, implement and demonstrate the Candidate Elimination ALgoirthm to output a description 
         of the set of all hypothesis consistent with the training examples. '''

import csv
data = []
print("\n The given training data set is- \n")

# Read dataset row-wise and add to the list data
with open('ENJOYSPORT.csv', 'r') as csvFile:
  reader = csv.reader(csvFile)
  for row in reader:
    data.append(row)
    print(row)
num_attributes = len(data[0]) - 1    #Last column is excluded

print("\n The initial value of hypothesis: ")
s = ['0'] * num_attributes
g = ['?'] * num_attributes
print("\n The most specific hypothesis s0 is: [0,0,0,0,0,0] \n")
print("\n The most general hypothesis g0 is: [?,?,?,?,?,?] \n")

#Compare with first training set
for j in range(0, num_attributes):
  s[j] = data[1][j];
  
#Comparing with remaining training examples of given data set
print("\n Candidate Elimination Algorithm Hypothesis Version Space Computation \n")
temp = []
for i in range(1, len(data)):
  print("-----------------------------------------------------------------")
  if data[i][num_attributes] == '1':
    for j in range(0, num_attributes):
      if data[i][j] != s[j]:
        s[j] = '?'
        
    for j in range(0, num_attributes):
      for k in range(1, len(temp)):
        if temp[k][j] != '?' and temp[k][j] != s[j]:
          del temp[k]
          
    print("For Training example no. {0} the hypothesis is S{0}".format(i),s)
    if(len(temp) == 0):
      print("For Training example no. {0} the hypothesis is G{0}".format(i),g)
    else:    
      print("For Training example no. {0} the hypothesis is G{0}".format(i),temp)
      
  if data[i][num_attributes] == '0':
    for j in range(0, num_attributes):
      if s[j] != data[i][j] and s[j] != '?':
        g[j] = s[j]
        temp.append(g)
        g = ['?'] * num_attributes
        
    print(" For Training Example no. {0} the hypothesis is S{0} ".format(i),s)
    print(" For Training Example no. {0} the hypothesis is G{0}".format(i),temp)  
