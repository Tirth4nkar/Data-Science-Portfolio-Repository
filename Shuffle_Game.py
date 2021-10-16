from random import shuffle

def shuffle_list(mylist):
    shuffle(mylist)
    return mylist

def user_guess():
    example=['','o','']
    print(f'Hey so here is our primary placements of {example} \n where the first item is at 0, second one at 1 & the final one at 2.you have to guss the position of O after we shuffle it.')
    guess=''                          
    index_list=['0','1','2']
    while guess not in index_list:
        guess=input("pick one number between 0, 1 & 2:")
    return int(guess)

def guess_check(mylist,guess):
    if mylist[guess]== 'o':
       return "well played!"
    else:
        return "better luck next time"
        
example=['','o','']               
#shuffling example list
shuffled_list=shuffle_list(example)
#take user input
user_input=user_guess()
#check the guess             
guess_check(shuffled_list, user_input)


   

