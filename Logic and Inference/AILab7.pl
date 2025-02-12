
to_num(Words, Number) :-
    split_string(Words, " ", "", WordList),
    maplist(atom_string, WordAtoms, WordList),
    parse_number(WordAtoms, Number).

% Base cases for individual numbers
word_to_number(zero, 0).
word_to_number(one, 1).
word_to_number(two, 2).
word_to_number(three, 3).
word_to_number(four, 4).
word_to_number(five, 5).
word_to_number(six, 6).
word_to_number(seven, 7).
word_to_number(eight, 8).
word_to_number(nine, 9).

% Base cases for teens
word_to_number(ten, 10).
word_to_number(eleven, 11).
word_to_number(twelve, 12).
word_to_number(thirteen, 13).
word_to_number(fourteen, 14).
word_to_number(fifteen, 15).
word_to_number(sixteen, 16).
word_to_number(seventeen, 17).
word_to_number(eighteen, 18).
word_to_number(nineteen, 19).

% Base cases for tens
word_to_number(twenty, 20).
word_to_number(thirty, 30).
word_to_number(forty, 40).
word_to_number(fifty, 50).
word_to_number(sixty, 60).
word_to_number(seventy, 70).
word_to_number(eighty, 80).
word_to_number(ninety, 90).

% Special case for one thousand
parse_number(['one', 'thousand'], 1000).

% Parse hundreds
parse_number([H, 'hundred'], Number) :-
    word_to_number(H, HNum),
    Number is HNum * 100.

parse_number([H, 'hundred', 'and' | Rest], Number) :-
    word_to_number(H, HNum),
    parse_number(Rest, RestNum),
    Number is HNum * 100 + RestNum.

% Parse tens and units
parse_number([T], Number) :-
    word_to_number(T, Number).

parse_number([T, U], Number) :-
    word_to_number(T, TNum),
    word_to_number(U, UNum),
    Number is TNum + UNum.

% Handle compound tens and units (e.g., twenty one)
parse_number([Tens, Units], Number) :-
    word_to_number(Tens, TensNum),
    word_to_number(Units, UnitsNum),
    Number is TensNum + UnitsNum.

% Parse the entire number
parse_number(Words, Number) :-
    append(TensWords, [Units], Words),
    word_to_number(Units, UnitsNum),
    parse_number(TensWords, TensNum),
    Number is TensNum + UnitsNum.

% Parse the entire number with "and"
parse_number(Words, Number) :-
    append(HundredWords, ['and' | Rest], Words),
    parse_number(HundredWords, HundredsNum),
    parse_number(Rest, RestNum),
    Number is HundredsNum + RestNum.
