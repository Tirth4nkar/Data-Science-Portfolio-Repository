from random import randint
print("WELCOME TO GUESS ME!")
print("I'm thinking of a number between 1 and 100")
print("If your guess is more than 10 away from my number, I'll tell you you're COLD")
print("If your guess is within 10 of my number, I'll tell you you're WARM")
print("If your guess is farther than your most recent guess, I'll say you're getting COLDER")
print("If your guess is closer than your most recent guess, I'll say you're getting WARMER")
print("LET'S PLAY!")
num=randint(1,100)
guess_list=[0]
while True:    
    guess=int(input("pick a number between 1 & 100 "))
    guess_list.append(guess)
    if guess<10 and guess>100:
        print("Out of Bounds! \n please try again")
    while guess in range(1,100):     
        if num==guess:
            print(f'Hola! You have hit the bulls eye in {len(guess_list)} no of guesses')
        break
        if abs(num-guess_list[-2])<abs(num-guess_list[-1]):
                print("Warmer!")
        else:
                print("Colder!")
        if abs(num-guess)<10:
            print("Warm")
        else:
            print("cold")



#end of code

    
