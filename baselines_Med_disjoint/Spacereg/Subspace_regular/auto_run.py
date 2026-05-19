from time import sleep
import datetime
import os

def check_task_over():
    fact_forward = "/hdd2/cil/running_base/fact_forward/fact/results/"

    file_path = fact_forward + "dice_step_0_200_test_newUNET_dynamic.txt"
    print(file_path)


    # file creation timestamp in float
    c_time = os.path.getctime(file_path)
    # convert creation timestamp into DateTime object
    dt_c = datetime.datetime.fromtimestamp(c_time)
    print('Created on:', dt_c)



    # file modification timestamp of a file
    m_time = os.path.getmtime(file_path)
    # convert timestamp into DateTime object
    dt_m = datetime.datetime.fromtimestamp(m_time)
    #print('Modified on:', dt_m)

    date_time_str = dt_m.strftime("%Y-%m-%d")       #dt_m.strftime("%Y-%m-%d %H:%M:%S")

    print('Modified on:', date_time_str)

    mod_date_ch = "2023-11-05"
    
    print("Current time ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("This process has the PID", os.getpid())

    if date_time_str >=mod_date_ch:
        print("fact process is over")
        return True
    else:
        print("fact is running")
        return False
        


def main():
    #sleep(5)
    while check_task_over() is False:
        sleep(5)
    
    print("Subprocess will initiate")
    #run_space_reg()
    
        
        

if __name__ == "__main__":
    main()

