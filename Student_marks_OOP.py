class Student:

    def get_details(self):
        self.roll_no=int(input("Enter Roll NO:)"))
        self.name=input("Enter Name:")
        
        print(self.roll_no,self.name)



    def calculate_result(self):

        self.m1=float(input("Enter marks One:"))
        self.m2=float(input("Enter marks Two:"))
        self.m3=float(input("Enter marks three:"))
        self.total=self.m1+self.m2+self.m3
        self.percent=(self.total*100)/300

        print(f"{self.percent}")

    def display_details(self):
        

        print(f"Roll_NO:{self.roll_no} and Name:{self.name} and Percentage:{self.percent}%")

    

s1=Student()

s1.get_details()
s1.calculate_result()
s1.display_details()
