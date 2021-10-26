import random

def choose_first():
    if random.randint(1,2)==2:
        return 'Player 2'
    else:
        return 'Player 1'


def game_algo(user_feedback,input_location_row,input_location_column):
    if input_location_row=="row1":
        if input_location_column==1:
            row1[0]=user_feedback
        elif input_location_column==2:
            row1[1]=user_feedback
        else:
            row1[-1]=user_feedback
    
    if input_location_row=="row2":
        if input_location_column==1:
            row2[0]=user_feedback
        elif input_location_column==2:
            row2[1]=user_feedback
        else:
            row2[-1]=user_feedback
    
    if input_location_row=="row3":
        if input_location_column==1:
            row3[0]=user_feedback
        elif input_location_column==2:
            row3[1]=user_feedback
        else:
            row3[-1]=user_feedback
    
    return display(row1,row2,row3)  

def replay():
     return input('Do you want to play again? Enter Yes or No(Y/N): ').upper()

def win_check(row1,row2,row3):
    if row1[0]==row1[1]==row1[2] or row2[0]==row2[1]==row2[2] or row3[0]==row3[1]==row3[2]:
        print("{} wins!".format(row1[0],row2[0],row3[0])) 
    elif row1[0]==row2[1]==row3[2] or row1[2]==row2[1]==row3[0]:
        print("{} wins!".format(row1[0],row1[2]))
    elif row1[0]==row2[0]==row3[0] or row1[1]==row2[1]==row3[1] or row1[2]==row2[2]==row3[2]:
        print("{} wins!".format(row1[0],row2[0],row3[0]))
    else:
        print("it's a draw!")
    

row1=['','','']
row2=['','','']
row3=['','','']
print("Welcome to Tic Tac Toe game")
print("we have a 3x3 matrix as our tic tac toe board.")
print(f'Here is our initial board {display(row1,row2,row3)}')
print("you have to pick cells using default matrix notations.")
print("there are three rows.")
print("The input must be a X or O")
play_game = input('Are you ready to play? Enter Yes or No(Y/N).').upper()
if play_game.upper()=='Y':
    game_on=True
else:
    game_on=False
choose_first()
while game_on:
    user_feedback=input("What is your input? X/O").upper()
    input_location_row=input("in which row you want your input to be? row1, row2 or row3?").lower()
    input_location_column=int(input("add the column"))
    game_algo(user_feedback,input_location_row,input_location_column)
win_check(row1, row2, row3)
if replay()==N:
    print("Goodbye!")
