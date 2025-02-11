RegAlg: Average.cpp makefile
	g++ -o GenChange GenChange.cpp -O3 -std=c++17
	g++ -o Instant Instant.cpp Parameters.cpp -O3 -std=c++17
	g++ -o Average Average.cpp Parameters.cpp -O3 -std=c++17
	g++ -o CCE_Average CCE_Average.cpp Parameters.cpp -O3 -std=c++17
	g++ -o CCE_Instant CCE_Instant.cpp Parameters.cpp -O3 -std=c++17

