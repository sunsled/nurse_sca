import optimizers.WOA as woa
import optimizers.GA as ga
import optimizers.SCA as sca


from pprint import pprint
import math
import itertools
import time
import datetime
import csv
import os
import numpy as np

'''
Let's assume nurse prefs are expressed as "aversions", scale of 1-4, where 4 is maximum aversion.
Let's assume for now that we are fulfilling a weekly schedule for full-time nurses.
We have 15 shifts per day: 6 day shifts, 6 afternoon shifts, and 3 night shifts.
15 shifts per day * 7 days = 105 shifts to fulfill per week.
If we assume each nurse works 5 shifts a week (40 hour work week),
Then we will observe 105/5 = 21 nurses to fill all shifts.
Our data will be 21 full-time nurses and 4 on-call nurses who will have no preference for shifts.
'''

'''
New constraints
Let's assume nurse prefs are expressed as "aversions", scale of 1-4, where 4 is maximum aversion.
Let's assume we have 5 nurses and 7 days.
We need to schedule 1 nurse per day.
Each nurse cannot work more than 2 shifts over 7 days.
Every nurse must work at least 1 shift.
No nurse may work adjacent shifts.
'''

#GLOBAL VALUES

#nurse preferences for working on a given day. nurses do not have individual shift preferences, only day preferences
prefs_input = [
    [4, 4, 2, 4, 1, 3, 2],
    [1, 1, 3, 4, 2, 1, 1],
    [2, 2, 3, 4, 4, 1, 1],
    [1, 3, 4, 2, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
]
#shifts
SHIFTS = 3

#how many tries we give our algorithm to find a viable solution
ATTEMPTS_ALLOWED = 3

#how many times we run the whole test to average out the randomness of our results
TEST_ITERATIONS = 5


#-----------------helper functions-------------- 
# functions which work as a black box, you give input you receive output

#returns the total aversion of a given assignment by matching it against given preferences
def calculationTotalAversion(prefs, assignment):
  HARD_HATE = len(prefs) * len(prefs[0]) * 20       # 20 is a lot more than 4 (4 is max hate)
  ans = 0

 
  # hard constraint: same nurse not on shift on the same day
  for day in range(len(assignment[0])):
    if assignment[0][day] == assignment[1][day]:
      ans += HARD_HATE * 10
    if assignment[1][day] == assignment[2][day]:
      ans += HARD_HATE * 10
    if assignment[0][day] == assignment[2][day]:
      ans += HARD_HATE * 10
  
  # Hard Constraint: cannot have a Day shift after a Night shift in the previous day
  for day in range(len(assignment[0])):
    if day != 0:
      if assignment[0][day] == assignment[2][day - 1]:
        ans += HARD_HATE * 10

  # hard constraint: no adjacent workdays
  # for shift in assignment:
  #   for day in range(1, len(shift)):
  #     if shift[day-1] == shift[day]:
  #       ans += HARD_HATE
  
      
  days_worked = [0] * len(prefs)
  for shift in assignment:
    for day in range(len(shift)):
      days_worked[shift[day]-1] += 1 

  # hard constraint: at most 5 workdays or at least 1 day
  for nurse in days_worked:
    if nurse > 5 or nurse < 1:
      ans += HARD_HATE

  # soft constraints: matches current schedule vs preferences
  nurse_ans = 0
  for shift in assignment:
      for i in range(len(shift)):
          nurse_ans += prefs_input[shift[i]-1][i]
  ans += nurse_ans
   

  return ans

#for GA
def calculationTotalAversionGA(prefs, assignment):
  HARD_HATE = len(prefs) * len(prefs[0]) * 20       # 20 is a lot more than 4 (4 is max hate)
  ans = 0

  # hard constraint: 2 nurses on duty today
  for day in range(len(prefs[0])):      
    num_on_duty = sum([assignment[nurse][day] for nurse in range(len(prefs))])
    if num_on_duty != 2:
      ans += HARD_HATE * 10

  for nurse in range(len(prefs)):
    nurse_ans = 0
    
    # hard constraint: at most 3 workdays
    # hard constraint: at least 2 workdays
    num_workdays = sum(assignment[nurse])
    if num_workdays > 3 or num_workdays < 2:
      nurse_ans += HARD_HATE

    for day in range(len(prefs[0])):     
      # hard constraint: no adjacent workdays
      if day != 0 and assignment[nurse][day] > 0.5 and assignment[nurse][day-1] > 0.5:
        nurse_ans += HARD_HATE


      # soft constraints: matches current schedule vs preferences
      nurse_ans += assignment[nurse][day] * prefs[nurse][day]
    ans += nurse_ans
  return ans


#changes output to something we can work with
def restructure(arr, stride):
  ans = []
  for index, element in enumerate(arr):
    if index % stride == 0:
      ans.append([])
    ans[-1].append(element)
  return ans


#takes matrix of results, outputs a csv file
def convertToCsv(results, test, algo):
  #results = total runs, tries, searchagents, iterations, time
  ct = datetime.datetime.now()
  ct = str(ct)[0:-10]
  ct = ct[:-6]+'-'+ct[-5:-3]+'-'+ct[-2:]

  #section to write results to new excel document    
  newSheetName = ct + '-' + test + '.csv'
  cwd = os.getcwd()
  newSheetName = os.path.join(cwd, 'datasheets', newSheetName)
  pprint(results)
  results = sorted(results)

  with open(newSheetName, 'w', newline='') as writeFile:
    writer = csv.writer(writeFile)
    #end_results = [searchagents*iterations, success_rate, average tries, searchagents, iterations, average time elapsed]
    results.insert(0, ['Total Runs', 'Success Rate', 'Average Tries', 'Search Agents', 'Iterations', 'Average Time to Find Solution', 'Average Time Elapsed Per WOA (~Avg Total/Tries)'])
    results.insert(0, [algo, 'Test'])


    for row in results:
        writer.writerow(row)
  writeFile.close()  

  print('Results written to', newSheetName)


#takes 3d array of results, joins every matrix and calculates average values
def formatResults(sum_results):
  
  formatted_res = [[0] * (len(sum_results[0][0])+1) for i in range(len(sum_results[0]))]

  for j in range(len(sum_results[0])):
      formatted_res[j][1] = sum(1 for matrix in sum_results if matrix[j][1] < ATTEMPTS_ALLOWED) / len(sum_results)
      formatted_res[j][2] = sum(matrix[j][1] for matrix in sum_results) / len(sum_results)
      formatted_res[j][5] = sum(matrix[j][4] for matrix in sum_results) / len(sum_results)

      formatted_res[j][0] = sum_results[0][j][0]
      formatted_res[j][3] = sum_results[0][j][2]
      formatted_res[j][4] = sum_results[0][j][3]
      formatted_res[j][6] = sum_results[0][j][5]

  return formatted_res

#-----------run optimization--------------
# functions that utilize the algorithms alongside helper functions to actually optimize our schedule

def doNurseOptimization(prefs, SearchAgents, Max_iter, optimizer_name='SCA'):

  # creates a benchmark function (called objf)
  def objf(x): 

    # floors every element, and then restructures into a rectangle    
    assignment = restructure([math.floor(thing) for thing in x], len(prefs[0]))


    return calculationTotalAversion(prefs, assignment)

  # sets up arguments for optimizer (dim, SearchAgents_no, Max_iter)
  NUM_SHIFTS = SHIFTS

  num_days = len(prefs[0])
  dim = NUM_SHIFTS * num_days
  SearchAgents_no = 20      # edit me
  Max_iter = 1000          # edit me

  # runs optimizer (to get answer)
  raw_woa_ans = sca.SCA(objf, 1, len(prefs_input), dim, SearchAgents_no, Max_iter)
  raw_woa_ans_vect = raw_woa_ans.bestIndividual

  print('raw output')
  print(raw_woa_ans_vect)
  '''
  for x in raw_woa_ans_vect:
    print(x)
  '''

  raw_woa_ans_vect = [math.floor(elt) for elt in raw_woa_ans_vect]
  woa_ans = restructure(raw_woa_ans_vect, len(prefs[0]))
 
 
  print('\nconverted output')
  for x in woa_ans:
    print(x)
  print('')
  return [woa_ans, raw_woa_ans.best]



def doNurseOptimizationGA(prefs, SearchAgents, Max_iter, optimizer_name='WOA'):

  # creates a benchmark function (called objf)
  def objf(x): 
    # rounds every element, and then restructures into a rectangle
    
    assignment = restructure([round(thing) for thing in x], len(prefs[0]))

    return calculationTotalAversionGA(prefs, assignment)

  # sets up arguments for optimizer (dim, SearchAgents_no, Max_iter)
  num_nurses = len(prefs)
  num_days = len(prefs[0])
  dim = num_nurses * num_days
  #SearchAgents_no = 30     # edit me
  #Max_iter = 50            # edit me

  # runs optimizer (to get answer)
  raw_woa_ans = ga.GA(objf, 0, 1, dim, SearchAgents, Max_iter)
  raw_woa_ans_vect = raw_woa_ans.bestIndividual

  #print(raw_woa_ans_vect)

  #print('raw_woa_ans_vect \n', raw_woa_ans_vect)
  raw_woa_ans_vect = [round(elt) for elt in raw_woa_ans_vect]
  woa_ans = restructure(raw_woa_ans_vect, len(prefs[0]))

  
  return [woa_ans, raw_woa_ans.best]


#----------tests-----------
# functions to test our algorithms and return data

def runTest2(prefs, searchagents, max_iter):
  #run until feasible solution is found, count how many tries it took
  feasible_solution = None
  iterations = 0
  sum_time = 0
  while not feasible_solution:
    start = time.time()
    solution = doNurseOptimization(prefs, searchagents, max_iter)[1]
    end = time.time()
    sum_time += (end-start)
    if solution < 700:
      feasible_solution = solution
    iterations += 1
    if iterations > ATTEMPTS_ALLOWED:
      iterations = ATTEMPTS_ALLOWED
      break
  avg_time = sum_time / iterations

  # [the most optimal schedule, how many tries it took, how long it took on average]
  return [solution, iterations, avg_time]
  


def runTest2GA(prefs, searchagents, max_iter):
  #run until feasible solution is found, count how many tries it took
  feasible_solution = None
  iterations = 0
  sum_time = 0

  while not feasible_solution:
    start = time.time()
    solution = doNurseOptimizationGA(prefs, searchagents, max_iter)[1]
    end = time.time()
    sum_time += (end-start)
    if solution < 700:
      feasible_solution = solution
    iterations += 1
    if iterations > ATTEMPTS_ALLOWED:
      iterations = ATTEMPTS_ALLOWED
      break
  avg_time = sum_time / iterations
  
  # [the most optimal schedule, how many tries it took, how long it took on average]
  return [solution, iterations, avg_time]



#outermost function, it runs our test cases
def testAlgo(doWOA, doGA):
  COMBINATIONS = [range(10,21,10), range(10,21,10)]  
  AGENT_ITER_LIST_TEMP = list(itertools.product(*COMBINATIONS))
  AGENT_ITER_LIST = []
  max_iteration = 400
  min_iteration = 400

  
  for i in range(len(AGENT_ITER_LIST_TEMP)):
    if AGENT_ITER_LIST_TEMP[i][0] * AGENT_ITER_LIST_TEMP[i][1] <= max_iteration and AGENT_ITER_LIST_TEMP[i][0] * AGENT_ITER_LIST_TEMP[i][1] >= min_iteration:
      AGENT_ITER_LIST.append(AGENT_ITER_LIST_TEMP[i])

  print(AGENT_ITER_LIST)
  length = len(AGENT_ITER_LIST)


  #WHAT IS SUCCESS RATE
  # testAlgo loops over TEST_ITERATIONS
  # each iteration calls test2
  # test 2 runs until it finds a solution or breaks at ATTEMPTS_ALLOWED
  # success rate is how many testAlgo iterations (aka test2 calls) that found a viable solution before breaking

  #test case using Whale Optimization Algorithm
  if doWOA:
    sum_results = []
    for iteration in range(TEST_ITERATIONS):
      results = []
      print('test iteration', iteration, 'of', TEST_ITERATIONS)
      for i in range(length):
        current_combination = AGENT_ITER_LIST[i]
        print('running test 2 with', str(current_combination), 'at iteration', str(i), 'of', str(length))
        start = time.time()
        test_result = runTest2(prefs_input, current_combination[0], current_combination[1])
        end = time.time()
        #total runs, tries, searchagents, iterations, time, avg_time
        results.append([current_combination[0] * current_combination[1], test_result[1], current_combination[0], current_combination[1], abs(end - start), test_result[2]])
        
      sum_results.append(results)

    end_results = formatResults(sum_results)

    convertToCsv(end_results, 'testWOA', 'WOA')

    print('WOA complete')

  #test case using Genetic Algorithm
  if doGA:
    sum_results = []
    for iteration in range(TEST_ITERATIONS):
      results = []
      print('test iteration', iteration, 'of', TEST_ITERATIONS)
      for i in range(length):
        current_combination = AGENT_ITER_LIST[i]
        print('running test 2 with', str(current_combination), 'at iteration', str(i), 'of', str(length))
        start = time.time()
        test_result = runTest2GA(prefs_input, current_combination[0], current_combination[1])
        end = time.time()
        results.append([current_combination[0] * current_combination[1], test_result[1], current_combination[0], current_combination[1], abs(end - start), test_result[2]])
        
      sum_results.append(results)

    end_results = formatResults(sum_results)

    convertToCsv(end_results, 'testGA', 'GA')


    print('GA complete')
  

#runs our tests
testAlgo(True, False)






#---------------------------------OLD STUFF-------------------------------------------

#boolean vs continuous problems
#integer problems vs real value problems

#possible that our problem can't be solved using WOA or a similar algorithm

#try to change what we want our output to be, to better match how WOA works
#woa likes to use float values rather than the boolean 0 and 1 we have assigned it

#some hard constraints are hardcoded to work for boolean values 0 and 1, if we change what our output looks like we need to change how the constraints are checked

'''

  if runOne:
    results = []
    TEST_ITERATIONS = 50
    for i in range(length):
      current_combination = AGENT_ITER_LIST[i]
      print('running test 1 with', str(current_combination), 'at iteration', str(i), 'of', str(length))
      start = time.time()
      test_result = runTest1(prefs_input, current_combination[0], current_combination[1], TEST_ITERATIONS)
      end = time.time()
      results.append([current_combination[0] * current_combination[1], str(test_result[1]*100) + '%', current_combination[0], current_combination[1], abs(end - start)])

    #sort results by total iterations
    
    #pprint(results)
    #print(results[np.argsort(results[:,0])])
    convertToCsv(results, 'test1', 'Viable Solutions')
'''



  
'''
def runTestsGA(pref, runOne, runTwo, runThree):
  COMBINATIONS = [range(10,1001, 10), range(10,1001, 10)]  
  AGENT_ITER_LIST_TEMP = list(itertools.product(*COMBINATIONS))
  AGENT_ITER_LIST = []
  #maybe iterate over list and remove outlier values at arbitrary number
  max_iteration = 2000
  min_iteration = 200

  for i in range(len(AGENT_ITER_LIST_TEMP)):
    if AGENT_ITER_LIST_TEMP[i][0] * AGENT_ITER_LIST_TEMP[i][1] <= max_iteration and AGENT_ITER_LIST_TEMP[i][0] * AGENT_ITER_LIST_TEMP[i][1] >= min_iteration:
      AGENT_ITER_LIST.append(AGENT_ITER_LIST_TEMP[i])

  length = len(AGENT_ITER_LIST)
  print(AGENT_ITER_LIST)

  if runOne:
    results = []
    TEST_ITERATIONS = 50
    for i in range(length):
      current_combination = AGENT_ITER_LIST[i]
      print('running test 1 genetic algorithm with', str(current_combination), 'at iteration', str(i), 'of', str(length))
      start = time.time()
      test_result = runTest1GA(prefs_input, current_combination[0], current_combination[1], TEST_ITERATIONS)
      end = time.time()
      results.append([current_combination[0] * current_combination[1], str(test_result[1]*100) + '%', current_combination[0], current_combination[1], abs(end - start)])

    #sort results by total iterations
    
    #pprint(results)
    #print(results[np.argsort(results[:,0])])
    convertToCsv(results, 'test1GA', 'Viable Solutions')

  #------------test 2----------------

  #results = [total runs, tries, search agents, iterations, time elapsed]
  if runTwo:
    results = []
    for i in range(length):
      current_combination = AGENT_ITER_LIST[i]
      print('running test 2 genetic algorithm  with', str(current_combination), 'at iteration', str(i), 'of', str(length))
      start = time.time()
      test_result = runTest2GA(prefs_input, current_combination[0], current_combination[1])
      end = time.time()
      results.append([current_combination[0] * current_combination[1], test_result[1], current_combination[0], current_combination[1], abs(end - start)])

    convertToCsv(results, 'test2GA', '# of Attempts')

  if runThree:
    pass

#--------run tests----------
'''
#runTests(prefs_input, True, False, False)
#runTestsGA(prefs_input, True, False, False)






'''
  AGENT_ITER_LIST = [
    #------------
    [5, 10],
    [5, 50],
    [5, 100],
    [5, 500],
    [5, 1000],
    [5, 3000],
    #
    [10, 5],
    [50, 5],
    [100, 5],
    [500, 5],
    [1000, 5],
    [3000, 5],
    #------------
    [15, 10],
    [15, 50],
    [15, 100],
    [15, 500],
    #
    [10, 15],
    [50, 15],
    [100, 15],
    [500, 15],
    #------------
    [50, 10],
    [50, 50],
    [50, 100],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
  ]
'''
'''
def runTests(pref, runOne, runTwo, runThree):
  
  #this takes too much runtime, just hardcode desirable values instead
  #every combination of searchagents and iterations from x-y skipping by z
  COMBINATIONS = [range(10,1001, 10), range(10,1001, 10)]  
  AGENT_ITER_LIST_TEMP = list(itertools.product(*COMBINATIONS))
  AGENT_ITER_LIST = []
  #maybe iterate over list and remove outlier values at arbitrary number
  max_iteration = 2000
  min_iteration = 200
  for i in range(len(AGENT_ITER_LIST_TEMP)):
    if AGENT_ITER_LIST_TEMP[i][0] * AGENT_ITER_LIST_TEMP[i][1] <= max_iteration and AGENT_ITER_LIST_TEMP[i][0] * AGENT_ITER_LIST_TEMP[i][1] >= min_iteration:
      AGENT_ITER_LIST.append(AGENT_ITER_LIST_TEMP[i])

  length = len(AGENT_ITER_LIST)
  print(AGENT_ITER_LIST)
  #every combination takes factorial runtime, just hardcode desirable values instead
 
  #------------test 1----------------
  #results = [[searchagents*iterations, percent_feasible, searchagents, iterations, time_elapsed]...]
  if runOne:
    results = []
    TEST_ITERATIONS = 50
    for i in range(length):
      current_combination = AGENT_ITER_LIST[i]
      print('running test 1 with', str(current_combination), 'at iteration', str(i), 'of', str(length))
      start = time.time()
      test_result = runTest1(prefs_input, current_combination[0], current_combination[1], TEST_ITERATIONS)
      end = time.time()
      results.append([current_combination[0] * current_combination[1], str(test_result[1]*100) + '%', current_combination[0], current_combination[1], abs(end - start)])

    #sort results by total iterations
    
    #pprint(results)
    #print(results[np.argsort(results[:,0])])
    convertToCsv(results, 'test1', 'Viable Solutions')

  #------------test 2----------------

  #results = [total runs, tries, search agents, iterations, time elapsed]
  if runTwo:
    results = []
    for i in range(length):
      current_combination = AGENT_ITER_LIST[i]
      print('running test 2 with', str(current_combination), 'at iteration', str(i), 'of', str(length))
      start = time.time()
      test_result = runTest2(prefs_input, current_combination[0], current_combination[1])
      end = time.time()
      results.append([current_combination[0] * current_combination[1], test_result[1], current_combination[0], current_combination[1], abs(end - start)])

    convertToCsv(results, 'test2', '# of Attempts')

  #------------test 3----------------
'''


'''
results = doNurseOptimization(prefs_input, 1000, 5)

for shift in results[0]:
  print(shift)

print(results[1])
'''
'''
def runTest1(prefs, searchagents, max_iter, test_iter):
  #run each set of values multiple times
  #calculate % accuracy (how many solutions are less than 700 aversion)

  scores = []
  num_feasible = 0
  for i in range(test_iter):
    score = doNurseOptimization(prefs, searchagents, max_iter)[1]
    if score < 700:
      num_feasible += 1
    scores.append(score)

  percent_feasible = num_feasible / len(scores)

  return [scores, percent_feasible]

#print(runTest1(prefs_input, 30, 30, 10))
'''



'''
  # old constraints
  # hard constraint: no nurses works 2 days in a row (ignoring wraparound)

  for day in range(1, len(assignment)):   #7 days in a week
    if assignment[day-1] == assignment[day]:
      ans += HARD_HATE
  

  #hard constraint:
  #  no nurse works more than 2 days
  #  every nurse works at least 1 day
  for nurse in range(len(prefs)):
    days_worked = 0
    for day in assignment:
      if day == nurse+1:
        days_worked+=1
    if days_worked < 1 or days_worked > 2:
      ans += HARD_HATE
  
  nurse_ans = 0
  for day in range(len(assignment)):
    nurse_ans += prefs[assignment[day]-1][day]

  ans += nurse_ans
  
  print('assignment in aversion ')
  for shift in assignment:
    print(shift)
  return 0

'''




'''
def runTest1GA(prefs, searchagents, max_iter, test_iter):
  #run each set of values multiple times
  #calculate % accuracy (how many solutions are less than 700 aversion)
  

  scores = []
  num_feasible = 0
  for i in range(test_iter):
    score = doNurseOptimizationGA(prefs, searchagents, max_iter)[1]
    if score < 700:
      num_feasible += 1
    scores.append(score)

  percent_feasible = num_feasible / len(scores)

  return [scores, percent_feasible]
'''


''' 
  sum_results
  [
    sum_results[i]
    [
      #[total runs, tries, searchagents, iterations, time]
      [100, 10, 10, 10, 0.6313107013702393], 
      [200, 5, 20, 10, 0.5595159530639648]],
    [

      [100, 8, 10, 10, 0.45081615447998047], 
      [200, 2, 20, 10, 0.2234022617340088]],
    [
      [100, 10, 10, 10, 0.6068825721740723], 
      [200, 1, 20, 10, 0.11021018028259277]
    ]
  ]

  =>

  [
    #[total runs, success_rate, average tries, searchagents, iterations, average time]
    [100, ignore this part, (tries+tries+tries)/3, 10, 10, (time+time+time)/3]
    [200, ignore this part, (tries+tries+tries)/3, 10, 10, (time+time+time)/3]
  ]

'''

'''
  pprint(sum_results)
  end_results = []
  for inner_length in range(len(sum_results[0])):

    temp = []
    #match rows from every result next to eachother
    for outer_length in (range(len(sum_results))):
        temp.append(sum_results[outer_length][inner_length])

    #create a format for calculating the average of rows
    new_temp = temp[0]
    if new_temp[1] >= ATTEMPTS_ALLOWED:  
      #new_temp = new_temp[0].append([0].append(new_temp[1:]))
      new_temp = [new_temp[0]] + [0] + new_temp[1:]
    else:
      #new_temp = new_temp[0].append([1].append(new_temp[1:]))
      new_temp = [new_temp[0]] + [1] + new_temp[1:]

    total = 1
    for i in range(1, len(temp)):
      
      print(new_temp)
      print(new_temp[2])
      print(temp[i])
      print(temp[i][2])
      
      new_temp[2] += temp[i][1]
      new_temp[5] += temp[i][4]
      if not (temp[i][1] >= ATTEMPTS_ALLOWED):
        new_temp[1] += 1

      total += 1


     
    #end_results = [searchagents*iterations, success_rate, average tries, searchagents, iterations, average time elapsed]
    #results = [total runs, tries, searchagents, iterations, time]

    new_temp[2] = str(round((new_temp[2] / total), 2))
    new_temp[5] = str(round((new_temp[5] / total), 2))
    end_results.append(new_temp)
    print('---')
    pprint(end_results)

  '''
    #return end_results
