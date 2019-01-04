
import random
previous_num=0
cnt = 0

for x in range(10):
  generated_number = random.randint(0,1)
  print (generated_number)
  


  #Counter

  if(previous_num < generated_number):
    print('Changed from 0 to 1')
    cnt = cnt+1
  previous_num = generated_number

print('Final count: ',cnt)

