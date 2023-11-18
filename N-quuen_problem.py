#Number of queens
print ("Enter the number of queens")
N = int(input())
#chessboard
#NxN matrix with all elements 0
#board = [[0]*N for _ in range(N)]
#print(4)
#print(board)
#print()
board = []
for _ in range(N):
    row = []
#    print(row)
#    print()
    for _ in range(N):
        row.append(0)
#        print(row)
    board.append(row)
print(board)
print()
#This code accomplishes the same task: it initializes an NxN matrix filled with zeros.
# The outer loop creates each row, and the inner loop populates each row with zeros.
# Adjust the value of N to create a matrix of the desired size.
#In simple terms, this code initializes a 2D list (board) with dimensions N by N and fills it with zeros.
# The outer loop (for _ in range(N)) iterates N times, creating a new row in each iteration. 
#The inner loop (for _ in range(N)) iterates N times for each row, adding zeros to the row.
# The resulting 2D list represents a matrix with N rows and N columns, where all elements are initially set to zero.
def is_attack(i, j):
    #checking if there is a queen in row or column
    for k in range(0,N):
        if board[i][k]==1 or board[k][j]==1:
            return True
    #checking diagonals
    for k in range(0,N):
        for l in range(0,N):
            if (k+l==i+j) or (k-l==i-j):
                if board[k][l]==1:
                    return True
    return False
#This code defines a function is_attack(i, j) that checks whether placing a queen at position (i, j) 
#on a chessboard is an "attack" or not. In chess, a queen can attack in the same row, column, or diagonally. 
#Let's break down the code in simpler terms:
#The first loop checks if there is a queen in the same row or column as the given position (i, j) by iterating through 
#the elements in that row and column.
#The second set of nested loops checks if there is a queen in any of the diagonals. 
#It does so by comparing the sum and difference of indices (k + l == i + j) and (k - l == i - j) 
#for all elements on the chessboard.
#If either condition is met (indicating an attack), the function returns True. 
#Otherwise, it returns False, indicating that placing a queen at the given position is safe.
#This function is commonly used in solving the N-Queens problem to check whether placing a queen at
# a particular position results in a conflict with existing queens on the board.
def N_queen(n):
    #if n is 0, solution found
    if n==0:
        return True
    for i in range(0,N):
        for j in range(0,N):
            '''checking if we can place a queen here or not
            queen will not be placed if the place is being attacked
            or already occupied'''
            if (not(is_attack(i,j))) and (board[i][j]!=1):
                board[i][j] = 1
                #recursion
                #wether we can put the next queen with this arrangment or not
                if N_queen(n-1)==True:
                    return True
                board[i][j] = 0

    return False

N_queen(N)
for i in board:
    print (i)