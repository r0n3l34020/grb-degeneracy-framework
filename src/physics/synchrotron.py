import numpy

flux_list = []

f_break = float(1.0)

nu_list = [10, 20, 100, 200]

p = float(input("What do you want the electron distribution index to be? "))

alpha_1 = (p-1)/2
alpha_2 = alpha_1 + 0.5

for i in nu_list:
    if i <= 50:
        result = f_break * (i / 50) ** (-alpha_1)
        round_result = round(result, 3)
        flux_list.append(round_result)
    
    else:
        result = f_break * (i / 50) ** (-alpha_2)
        round_result = round(result, 3)
        flux_list.append(round_result)

print(flux_list)