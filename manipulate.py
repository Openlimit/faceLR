import numpy as np
import struct
import sys
import re

transform = np.array([
    [-1.25, 0, 0, -15.4],
    [0, 1.25, 0, 0],
    [0, 0, -1.25, -1322],
    [0, 0, 0, 1]
])


def transform_obj(name):
    with open(name, mode='r') as f:
        with open(name[:-4] + '_tr.obj', mode='w') as wf:
            for i in range(2455):
                line = f.readline()
                strs = line.split(' ')
                x = float(strs[1])
                y = float(strs[2])
                z = float(strs[3])
                point = np.array([x, y, z, 1])
                tr_point = transform.dot(point)
                wf.write('v %f %f %f\n' % (tr_point[0], tr_point[1], tr_point[2]))
            wf.write(f.read())


def transform_marks(name):
    with open(name, mode='r') as f:
        with open(name[:-4] + '_tr.txt', mode='w') as wf:
            num = int(f.readline())
            wf.write('%d\n' % num)
            for i in range(num):
                line = f.readline()
                strs = line.split(' ')
                x = float(strs[0])
                y = float(strs[1])
                z = float(strs[2])
                point = np.array([x, y, z, 1])
                tr_point = transform.dot(point)
                wf.write('%f %f %f\n' % (tr_point[0], tr_point[1], tr_point[2]))


def dc(name):
    points = []
    with open(name, mode='r') as rf:
        num = int(rf.readline())
        with open(name[:-4] + '_dc.txt', mode='w') as wf:
            wf.write('%d\n' % (2 * num - 1))
            for i in range(num):
                line = rf.readline()
                wf.write(line)
                strs = line.split(' ')
                x = float(strs[0])
                y = float(strs[1])
                z = float(strs[2])
                points.append([x, y, z])
            s = points[num - 1][0]
            for i in reversed(range(num - 1)):
                wf.write('%f %f %f\n' % (2 * s - points[i][0], points[i][1], points[i][2]))


def bin2obj(name):
    with open(name, mode='rb') as f:
        with open(name[:-4] + '.obj', mode='w') as wf:
            vm = struct.unpack('i', f.read(4))[0]
            fm = struct.unpack('i', f.read(4))[0]
            for i in range(vm):
                x = struct.unpack('f', f.read(4))[0]
                y = struct.unpack('f', f.read(4))[0]
                z = struct.unpack('f', f.read(4))[0]
                wf.write('v %f %f %f\n' % (x, y, z))
            for i in range(vm):
                x = struct.unpack('f', f.read(4))[0]
                y = struct.unpack('f', f.read(4))[0]
                z = struct.unpack('f', f.read(4))[0]
                wf.write('vn %f %f %f\n' % (x, y, z))
                wf.write('vn 0 0 0\n')
            for i in range(fm):
                x = struct.unpack('i', f.read(4))[0] + 1
                y = struct.unpack('i', f.read(4))[0] + 1
                z = struct.unpack('i', f.read(4))[0] + 1
                wf.write('f %d/%d %d/%d %d/%d\n' % (x, x, y, y, z, z))


def obj2bin(name):
    with open(name, mode='r') as f:
        with open(name[:-4] + '.bin', mode='wb') as wf:
            vb = struct.pack('i', 2455)
            fb = struct.pack('i', 4434)
            wf.write(bytes(vb))
            wf.write(bytes(fb))
            for i in range(2455):
                line = f.readline()
                strs = line.split(' ')
                x = float(strs[1])
                y = float(strs[2])
                z = float(strs[3])
                wf.write(bytes(struct.pack('f', x)))
                wf.write(bytes(struct.pack('f', y)))
                wf.write(bytes(struct.pack('f', z)))
            for i in range(2455):
                f.readline()
            for i in range(4434):
                line = f.readline()
                strs = line.split(' ')
                x = int(strs[1].split('/')[0]) - 1
                y = int(strs[2].split('/')[0]) - 1
                z = int(strs[3].split('/')[0]) - 1
                wf.write(bytes(struct.pack('i', x)))
                wf.write(bytes(struct.pack('i', y)))
                wf.write(bytes(struct.pack('i', z)))
                if i == 4433:
                    print(line)


def marks2pp(name):
    with open(name, mode='r') as rf:
        vm = int(rf.readline())
        with open(name[:-4] + '.pp', mode='w') as wf:
            wf.write('<!DOCTYPE PickedPoints>\n<PickedPoints>\n')
            for i in range(vm):
                line = rf.readline()
                strs = line.split(' ')
                x = float(strs[0])
                y = float(strs[1])
                z = float(strs[2])
                wf.write('<point x="%f" y="%f" z="%f" active="1" name="%d"/>\n' % (x, y, z, i))
            wf.write('</PickedPoints>')


def pp2marks(name, n=77):
    dic = {}
    with open(name, mode='r') as rf:
        with open(name[:-3] + '.txt', mode='w') as wf:
            wf.write('%d\n' % n)
            xp = re.compile('x="(.*?)"')
            yp = re.compile('y="(.*?)"')
            zp = re.compile('z="(.*?)"')
            np = re.compile('name="(.*?)"')
            for i in range(n):
                line = rf.readline()
                x = float(xp.search(line)[1])
                z = float(zp.search(line)[1])
                y = float(yp.search(line)[1])
                index = int(np.search(line)[1])
                dic[index] = [x, y, z]
            for i in range(n):
                x, y, z = dic[i]
                wf.write('%f %f %f\n' % (x, y, z))


def gen_allmarks(old_marks, obj):
    with open(old_marks, mode='r') as old_rf:
        with open(obj, mode='r') as obj_rf:
            old_rf.readline()
            with open(old_marks[:-4] + '_all.txt', mode='w') as wf:
                wf.write('77\n')
                for i in range(15):
                    wf.write(old_rf.readline())
                dic = {}
                for i in range(60):
                    strs = obj_rf.readline().split(' ')
                    x = float(strs[1])
                    y = float(strs[2])
                    z = float(strs[3])
                    dic[i + 15] = [x, y, z]
                for i in range(60):
                    j = i + 15
                    real_index = j
                    if 15 <= j <= 20:
                        real_index = j + 6
                    elif 21 <= j <= 26:
                        real_index = j - 6
                    elif 27 <= j <= 30:
                        real_index = j + 4
                    elif 31 <= j <= 34:
                        real_index = j - 4
                    elif 35 <= j <= 43:
                        real_index = 2 * 39 - j
                    elif j == 44:
                        real_index = 45
                    elif j == 45:
                        real_index = 44
                    elif 46 <= j <= 52:
                        real_index = 2 * 49 - j
                    elif 53 <= j <= 57:
                        real_index = 2 * 55 - j
                    elif 58 <= j <= 60:
                        real_index = 2 * 59 - j
                    elif 61 <= j <= 63:
                        real_index = 2 * 62 - j
                    elif 65 <= j <= 72:
                        real_index = 137 - j
                    elif j == 73:
                        real_index = 74
                    elif j == 74:
                        real_index = 73
                    x, y, z = dic[real_index]
                    wf.write('%f %f %f\n' % (x, y, z))
                wf.write(old_rf.read())


if __name__ == '__main__':
    # bin2obj('/home/meidai/下载/im_std/stdhead.bin')
    # pp2marks('/home/meidai/下载/im_std/stdhead_picked_points.pp', n=9)
    # dc('/home/meidai/下载/im_std/stdhead_picked_points.txt')
    # marks2pp('/home/meidai/下载/im_std/stdhead_picked_points.txt')

    # transform_obj('/home/meidai/下载/im_std/stdhead.obj')
    # transform_marks('/home/meidai/下载/im_std/stdhead_picked_points.txt')
    # marks2pp('/home/meidai/下载/im_std/stdhead_picked_points_tr.txt')
    # obj2bin('/home/meidai/下载/im_std/stdhead_tr.obj')
    # gen_allmarks('/home/meidai/下载/stdhead_tr/marks_tr.txt', '/home/meidai/下载/stdhead_tr/stdhead_tr.obj')
    # marks2pp('/home/meidai/下载/stdhead_tr/marks_tr_all.txt')

    bin2obj('/home/meidai/下载/genobj_tym/frame_dump.bin')
    bin2obj('/home/meidai/下载/genobj_tym/legs_dump_left.bin')
    bin2obj('/home/meidai/下载/genobj_tym/legs_dump_right.bin')
    bin2obj('/home/meidai/下载/genobj_tym/lens_dump_left.bin')
    bin2obj('/home/meidai/下载/genobj_tym/lens_dump_right.bin')
