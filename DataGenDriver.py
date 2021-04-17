import os
import shutil
import subprocess

if __name__ == '__main__':

    # y_z_axis_angle_list = [[75.0, 345.0], [75.0, 330.0], [75.0, 315.0], [75.0, 300.0], [75.0, 285],
    #                        [60.0, 0.0], [60.0, 345.0], [60.0, 330.0], [60.0, 315.0], [60.0, 300.0], [60.0, 285.0], [60.0, 270.0],
    #                        [45.0, 345.0], [45.0, 330.0], [45.0, 300.0], [45.0, 285.0],
    #                        [30.0, 0.0], [30.0, 345.0], [30.0, 330.0], [30.0, 315.0], [30.0, 300.0], [30.0, 285.0], [30.0, 270.0],
    #                        [15.0, 0.0], [15.0, 345.0], [15.0, 330.0], [15.0, 315.0], [15.0, 300.0], [15.0, 285.0], [15.0, 270.0]]

    y_z_axis_angle_list = [[88.2, 321.8]]
    c_cnt = len(y_z_axis_angle_list)

    for idx in range(c_cnt):
        y_angle_str = str(y_z_axis_angle_list[idx][0])
        z_angle_str = str(y_z_axis_angle_list[idx][1])
        cmd = ['python3', 'DataGenerator.py', y_angle_str, z_angle_str]
        subprocess.Popen(cmd).wait()
        output_zip_name = 'SimData/TrainingData_IrreB_' + y_angle_str + '_' + z_angle_str + '_980'
        shutil.make_archive(output_zip_name, 'zip', 'SimData/TrainingData')
        c_cnt -= 1
        print("Rest cases:", c_cnt)


