import sys
from email.mime.text import MIMEText
from email.header import Header

import schedule
import smtplib
import pynvml
import os


def send_mail(msg):
    # 用于构建邮件头
    # 发信方的信息：发信邮箱，邮箱授权码
    # 邮箱的名称
    from_addr = '18093921686@163.com'
    # 邮箱的授权码
    # 参考 https://jingyan.baidu.com/article/495ba841ecc72c38b30ede38.html
    password = 'LRKTBPRKOGKSWBAS'

    # 收信方邮箱
    to_addr = '1510998508@qq.com'

    # 发信服务器
    smtp_server = 'smtp.163.com'

    # 邮箱正文内容，第一个参数为内容，第二个参数为格式(plain 为纯文本)，第三个参数为编码
    msg = MIMEText(msg, 'plain', 'utf-8')

    # 邮件头信息
    msg['From'] = Header(from_addr)
    msg['To'] = Header(to_addr)
    msg['Subject'] = Header('V100服务器GPU使用情况')

    # 开启发信服务，这里使用的是加密传输
    server = smtplib.SMTP_SSL(smtp_server)
    server.connect(smtp_server, 465)
    # 登录发信邮箱
    server.login(from_addr, password)
    # 发送邮件
    server.sendmail(from_addr, to_addr, msg.as_string())
    # 关闭服务器
    server.quit()


def look_gpu_info():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    free = meminfo.free >> 30
    flag = True
    # 如果空闲显存大于5G则发送邮件和运行代码
    if free >= 12:
        msg = 'GPU可用显存{}GB,立即查看Screen 是否正常执行程序！'.format(free)
        print(msg)

        # 运行py文件
        # os.system('python train.py')
        print("运行程序")
        # py文件中import xx或者 from xx import xxx时不能使用相对路径（如.packagename)
        # 运行py文件
        print("程序开始执行，停止监听！")
        os.system('python train.py')
        # 需要修改send_mail 方法中的一些配置信息
        send_mail(msg + "\n等待程序开始执行，监听任务结束！")
        # 取消定时任务
        schedule.clear()

        # 保证代码只执行一次
        flag = False

    return flag


# 需要install pynvml和schedule这两个包
if __name__ == '__main__':
    # 每五分钟运行一次
    # schedule.every(5).minutes.do(look_gpu_info)
    # 获取返回值 使得代码执行后不再监听
    job = schedule.every(1).seconds
    job.job_func = look_gpu_info
    schedule.jobs.append(job)
    result = job.run()
    print(result)

    while result:
        schedule.run_pending()

    schedule.clear()
    # sys.exit(0)
