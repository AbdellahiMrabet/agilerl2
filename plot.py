import math
import matplotlib.pyplot as plt
import csv
import numpy as np

x = []
y = []
t = []
z = []
z1 = []
z2 = []
z3 = []
y1 = []
y2 = []
y3 = []
t1 = []
t2 = []
t3 = []
x_tain = []
y_tain = []
z_tain = []
di = []
sd = []
didto = []
sddto = []
x_di = [5, 10, 15]
with open('./csv/evaldto_score_N9.csv', 'r') as file:
	plots = csv.reader(file, delimiter = ',')
	for row in plots:
		x.append(float(row[0]))
		y.append(float(row[1]))
		y1.append(float(row[2]))
		y2.append(float(row[3]))
		#y3.append(float(row[4]))

with open('./csv/evalMADDPG_score_N9.csv', 'r') as file0:
	plots = csv.reader(file0, delimiter = ',')
	for row in plots:
		z.append(float(row[1]))
		z1.append(float(row[2]))
		z2.append(float(row[3]))

# with open('singleagent.csv', 'r') as file1:
# 	plots = csv.reader(file1, delimiter = ',')
# 	for row in plots:
# 		z.append(float(row[1]))
# 		z1.append(float(row[2]))
# 		z2.append(float(row[3]))
# 		z3.append(float(row[4]))
  
with open('./csv/evalMATD3_score_N9.csv', 'r') as file3:
	plots3 = csv.reader(file3, delimiter = ',')
	for row in plots3:
		t.append(float(row[1]))
		t1.append(float(row[2]))
		t2.append(float(row[3]))
		t3.append(float(row[4]))
  
# with open('./csv/eval_fog_N_4.csv', 'r') as file4:
# 	plots = csv.reader(file4, delimiter = ',')
# 	su_values = []
# 	su = 0
# 	for row in plots:
# 		su += float(row[1])
# 		su_values.append(float(row[1]))
	
# 	average = np.mean(np.array(su_values))
# 	max = np.max(np.array(su_values))
# 	min = np.min(np.array(su_values))
# 	di.append((max - min) / average)
# 	sd.append(math.sqrt(su - average) / 5)
  
# with open('./csv/eval_fog_N_9.csv', 'r') as file5:
# 	plots = csv.reader(file5, delimiter = ',')
# 	su_values = []
# 	su = 0
# 	for row in plots:
# 		su += float(row[1])
# 		su_values.append(float(row[1]))
	
# 	average = np.mean(np.array(su_values))
# 	max = np.max(np.array(su_values))
# 	min = np.min(np.array(su_values))
# 	di.append((max - min) / average)
# 	sd.append(math.sqrt(su - average) / 10)
  
# with open('./csv/eval_fog_N_14.csv', 'r') as file6:
# 	plots = csv.reader(file6, delimiter = ',')
# 	su_values = []
# 	su = 0
# 	for row in plots:
# 		su += float(row[1])
# 		su_values.append(float(row[1]))
	
# 	average = np.mean(np.array(su_values))
# 	max = np.max(np.array(su_values))
# 	min = np.min(np.array(su_values))
# 	di.append((max - min) / average)
# 	sd.append(math.sqrt(su - average) / 15)
  
# with open('./csv/evaldto_fog_N_4.csv', 'r') as file2:
# 	plots = csv.reader(file2, delimiter = ',')
# 	su_values = []
# 	su = 0
# 	for row in plots:
# 		su += float(row[1])
# 		su_values.append(float(row[1]))
	
# 	average = np.mean(np.array(su_values))
# 	max = np.max(np.array(su_values))
# 	min = np.min(np.array(su_values))
# 	didto.append((max - min) / average)
# 	sddto.append(math.sqrt(su - average) / 5)
  
# with open('./csv/evaldto_fog_N_9.csv', 'r') as file7:
# 	plots = csv.reader(file7, delimiter = ',')
# 	su_values = []
# 	su = 0
# 	for row in plots:
# 		su += float(row[1])
# 		su_values.append(float(row[1]))
	
# 	average = np.mean(np.array(su_values))
# 	max = np.max(np.array(su_values))
# 	min = np.min(np.array(su_values))
# 	didto.append((max - min) / average)
# 	sddto.append(math.sqrt(su - average) / 10)
  
# with open('./csv/evaldto_fog_N_14.csv', 'r') as file8:
# 	plots = csv.reader(file8, delimiter = ',')
# 	su_values = []
# 	su = 0
# 	for row in plots:
# 		su += float(row[1])
# 		su_values.append(float(row[1]))
	
# 	average = np.mean(np.array(su_values))
# 	max = np.max(np.array(su_values))
# 	min = np.min(np.array(su_values))
# 	didto.append((max - min) / average)
# 	sddto.append(math.sqrt(su - average) / 15)
	
file.close()
file0.close()
#file1.close()
#file2.close()
file3.close()
# file4.close()
# file5.close()
# file6.close()
# file7.close()
# file8.close()		
#print("ration between two approaches: ", sum1 / sum2)
plt.plot(x, y, marker='o', label = r'$\epsilon$DTO')
plt.plot(x, z, marker='o', label = 'MADDPG')
plt.plot(x, t, marker='o', label = 'MADRL-MOO')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.grid()
#plt.xticks(range(1, len(x)+1))
plt.legend()
plt.savefig('reward_comparison.png')
plt.show()

plt.figure()
plt.plot(x, y1, marker='o', label = r'$\epsilon$DTO')
plt.plot(x, z1, marker='o', label = 'MADDPG')
plt.plot(x, t1, marker='o', label = 'MADRL-MOO')
plt.xlabel('Episode')
plt.ylabel('Exection Time')
plt.grid()
#plt.xticks(range(1, len(x)+1))
plt.legend()
plt.savefig('exection_time_comparison.png')
plt.show()

plt.figure()
plt.plot(x, y2, marker='o', label = r'$\epsilon$DTO')
plt.plot(x, z2, marker='o', label = 'MADDPG')
plt.plot(x, t2, marker='o', label = 'MADRL-MOO')
plt.xlabel('Episode')
plt.ylabel('Energy Consumption')
plt.grid()
#plt.xticks(range(1, len(x)+1))
plt.legend()
plt.savefig('energy_comparison.png')
plt.show()	

# plt.figure()
# plt.plot(x, y3, marker='o', label = r'$\epsilon$DTO')
# #plt.plot(x, z3, marker='o', label = 'MATD3-Wo-HPO')
# plt.plot(x, t3, marker='o', label = 'MADRL-MOO')
# plt.xlabel('Episode')
# plt.ylabel('Payment Cost')
# plt.grid()
# #plt.xticks(range(1, len(x)+1))
# plt.legend()
# plt.savefig('cost_comparison.png')
# plt.show()				

# plt.figure()
# plt.plot(x_di, didto, marker='o', label=r'$\epsilon$DTO')
# plt.plot(x_di, di, marker='o', label='MADRL-MOO')
# plt.xlabel('IoT number')
# plt.ylabel('DI')
# plt.grid()
# plt.legend()
# plt.savefig('degree_of_imbalance.png')
# plt.show()

# plt.figure()
# plt.plot(x_di, sddto, marker='o', label=r'$\epsilon$DTO')
# plt.plot(x_di, sd, marker='o', label='MADRL-MOO')
# plt.xlabel('IoT number')
# plt.ylabel('SD')
# plt.grid()
# plt.legend()
# plt.savefig('standard_deviation.png')
# plt.show()

# plt.figure()
# plt.plot(x_tain, y_tain, marker='o', label='MATD3-Wo-HPO')
# plt.plot(x_tain, z_tain, marker='o', label='MATD3-MOO')
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# #plt.grid()
# #plt.xticks(range(1, len(x)+1))
# plt.legend()
# plt.savefig('training_reward_wohpo_n9.png')
# plt.show()