# Machine Learning Classification with SKLearn

* We are going to predict net income based on one's life style
* The sample data may not be very accurate, it's generated randomly 
* The data looks as shown
* netSalary:lifeComfortable:hasCar:houseType:isBrokeEndMonth
* 160,000  : yes           : yes  :3 room   : no
* The model will help us predict the net pay give the above info
* Sample out put isshown below
* [0,1,0,1] - life comfort:yes,has car:no,broke end month:yes, house room: 1 room -> [9,000]
* [0,0,1,2] - life comfort:yes,has car:yes,broke end month: no, house room: 2 room -> [150,000]
* The 0's and 1's bra bra are used in data cleaning
* 0 means yes and 1 means no
* For house type it's different, 0 means 1 room, 1 means 2 room and so on...
* 
* Soon i will upload a graphical representation of the data 