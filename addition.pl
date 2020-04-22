nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).

%addition(X,Y) :- digit(X,Y).

addition(X,Z) :- digit(X,Y), W is Y+10, A is ((W-(W mod 10)) div 10), B = ((W mod 10)), Z is A+B.