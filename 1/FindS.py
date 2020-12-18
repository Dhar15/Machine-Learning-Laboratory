''' Q.1) Implement and demonstrate the FIND-S Algorithm for finding the most specific hypothesis based on a given set of training data samples. 
         Read the training data from a .CSV file. '''

import csv

def read_data(filename):
  with open(filename,"r") as csvfile:
    datareader = csv.reader(csvfile,delimiter=",")
    traindata = []
    for row in datareader:
      traindata.append(row)
    return (traindata)
  
#Function to find maximally specific set
def findS():
  dataarr = read_data('ENJOYSPORT.csv')
  hypothesis = ['0','0','0','0','0','0']
  rows = len(dataarr)
  columns = 7
  
  for x in range(1,rows):
    data = dataarr[x]
    print("Sample:", data)
    if data[columns-1] == '1':
      for y in range(0,columns-1):
        if hypothesis[y] == data[y]:
          pass
        elif hypothesis[y] != data[y] and hypothesis[y] == '0':
          hypothesis[y] = data[y]
        elif hypothesis[y] != data[y] and hypothesis[y] != '0':
          hypothesis[y] = '?'
      print("h0:", hypothesis)
      
  print("Maximally Specific Set")
  print('<',end=' ')
  for i in range(0,len(hypothesis)):
    print(hypothesis[i],',',end=' ')
  print('>')
  
findS()
