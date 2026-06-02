
"""
CPU-only, multiprocessing-optimized version of the COINS stroke generator.

Notes:
- No GPU / CuPy.
- Uses Pool(initializer=...) to share large read-only structures with worker processes,
  reducing per-task pickling overhead.
- Prefer running on a High-RAM Colab runtime for large shapefiles.
"""

import os, sys, math, time, multiprocessing
from functools import partial
import numpy as np
import shapefile as shp
import glob

# Increase recursion limit
sys.setrecursionlimit(20000)

# Globals that worker processes will use (set via initializer)
WORKER_TEMPARRAY = None
WORKER_UNIQUEDICT = None

def _init_worker(temp_array, unique_dict):
    """
    Pool initializer: set global references inside each worker process.
    This avoids pickling temp_array/unique_dict for every task.
    """
    global WORKER_TEMPARRAY, WORKER_UNIQUEDICT
    WORKER_TEMPARRAY = temp_array
    WORKER_UNIQUEDICT = unique_dict

# ---------- Utility functions (unchanged logic) ----------
def tupleToList(line):
    for a in range(0,len(line)):
        line[a] = list(line[a])
    return(line)

def listToTuple(line):
    for a in range(0, len(line)):
        line[a] = tuple(line[a])
    return(tuple(line))

def roundCoordinates(edge, decimal=4):
    x, y = edge
    return(round(x, decimal), round(y, decimal))

def listToPairs(inList):
    outList = []
    for index in range(0,len(inList)-1):
        tempList = [list(roundCoordinates(inList[index])), list(roundCoordinates(inList[index+1]))]
        outList.append(tempList)
    return(outList)

def computeAngle(point1, point2):
    # protect against zero division (degenerate points)
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    if dx == 0:
        return(90.0)
    angle = round(math.degrees(math.atan(abs(dy)/abs(dx))), 3)
    return(angle)

def computeOrientation(line):
    point1 = line[1]
    point2 = line[0]
    if ((point2[0] > point1[0]) and (point2[1] < point1[1])) or ((point2[0] < point1[0]) and (point2[1] > point1[1])):
        return(-computeAngle(point1, point2))
    elif point2[1] == point1[1]:
        return(0)
    elif point2[0] == point1[0]:
        return(90)
    else:
        return(computeAngle(point1, point2))

def pointsSetAngle(line1, line2):
    l1orien = computeOrientation(line1)
    l2orien = computeOrientation(line2)
    if ((l1orien>0) and (l2orien<0)) or ((l1orien<0) and (l2orien>0)):
        return(abs(l1orien)+abs(l2orien))
    elif ((l1orien>0) and (l2orien>0)) or ((l1orien<0) and (l2orien<0)):
        theta1 = abs(l1orien) + 180 - abs(l2orien)
        theta2 = abs(l2orien) + 180 - abs(l1orien)
        return(theta1 if theta1 < theta2 else theta2)
    elif (l1orien==0) or (l2orien==0):
        if l1orien<0:
            return(180-abs(l1orien))
        elif l2orien<0:
            return(180-abs(l2orien))
        else:
            return(180 - (abs(computeOrientation(line1)) + abs(computeOrientation(line2))))
    elif (l1orien==l2orien):
        return(180)

def angleBetweenTwoLines(line1, line2):
    l1p1, l1p2 = line1
    l2p1, l2p2 = line2
    l1orien = computeOrientation(line1)
    l2orien = computeOrientation(line2)
    if (l1orien==l2orien):
        angle = 180
    elif (l1orien==0) or (l2orien==0):
        angle = pointsSetAngle(line1, line2)
    elif l1p1 == l2p1:
        if ((l1p1[1] > l1p2[1]) and (l1p1[1] > l2p2[1])) or ((l1p1[1] < l1p2[1]) and (l1p1[1] < l2p2[1])):
            angle = 180 - (abs(l1orien) + abs(l2orien))
        else:
            angle = pointsSetAngle([l1p1, l1p2], [l2p1,l2p2])
    elif l1p1 == l2p2:
        if ((l1p1[1] > l2p1[1]) and (l1p1[1] > l1p2[1])) or ((l1p1[1] < l2p1[1]) and (l1p1[1] < l1p2[1])):
            angle = 180 - (abs(l1orien) + abs(l2orien))
        else:
            angle = pointsSetAngle([l1p1, l1p2], [l2p2,l2p1])
    elif l1p2 == l2p1:
        if ((l1p2[1] > l1p1[1]) and (l1p2[1] > l2p2[1])) or ((l1p2[1] < l1p1[1]) and (l1p2[1] < l2p2[1])):
            angle = 180 - (abs(l1orien) + abs(l2orien))
        else:
            angle = pointsSetAngle([l1p2, l1p1], [l2p1,l2p2])
    elif l1p2 == l2p2:
        if ((l1p2[1] > l1p1[1]) and (l1p2[1] > l2p1[1])) or ((l1p2[1] < l1p1[1]) and (l1p2[1] < l2p1[1])):
            angle = 180 - (abs(l1orien) + abs(l2orien))
        else:
            angle = pointsSetAngle([l1p2, l1p1], [l2p2,l2p1])
    return(angle)

# ---------- Multiprocessing worker functions now use globals set by initializer ----------
def getLinksMultiprocessing_worker(n_total):
    """
    Worker wrapper used in pool.map. Receives a single int (index).
    Uses global WORKER_TEMPARRAY.
    Returns (n, links_at_p1_list, links_at_p2_list)
    """
    n, total = n_total
    global WORKER_TEMPARRAY
    # progress
    if n % 1000 == 0:
        currentProgress = math.floor(100*n/total/2)
        remainingProgress = 50 - currentProgress
        print('>'*currentProgress + '-' * remainingProgress + ' [%d/%d] '%(n,total) + '%d%%'%(currentProgress*2), end='\r')

    tempArray = WORKER_TEMPARRAY
    # boolean masks (tempArray columns: [id, endpoint1_str, endpoint2_str])
    m1 = tempArray[:,1] == tempArray[n,1]
    m2 = tempArray[:,2] == tempArray[n,1]
    mask1 = np.logical_or(m1, m2)

    m1 = tempArray[:,1] == tempArray[n,2]
    m2 = tempArray[:,2] == tempArray[n,2]
    mask2 = np.logical_or(m1, m2)

    # extract ids as numpy integer array
    ids1 = tempArray[:,0][mask1].astype(np.int64)
    ids2 = tempArray[:,0][mask2].astype(np.int64)

    # filter out self
    ids1 = ids1[ids1 != n]
    ids2 = ids2[ids2 != n]

    return (n, ids1.tolist(), ids2.tolist())

def mergeLinesMultiprocessing_worker(n_total):
    """
    Worker wrapper for merge step that uses WORKER_UNIQUEDICT.
    Input: (index, total)
    Returns: sorted list of connected edges (as in original function)
    """
    n, total = n_total
    global WORKER_UNIQUEDICT
    if n % 1000 == 0:
        currentProgress = math.floor(100*n/total/2)
        remainingProgress = 50 - currentProgress
        print('>'*currentProgress + '-' * remainingProgress + ' [%d/%d] '%(n,total) + '%d%%'%(currentProgress*2), end='\r')

    uniqueDict = WORKER_UNIQUEDICT
    outlist = set()
    currentEdge1 = n
    outlist.add(currentEdge1)

    # forward walk
    while True:
        next_edge = uniqueDict[currentEdge1][6]
        if type(next_edge) == int and next_edge not in outlist:
            currentEdge1 = next_edge
            outlist.add(currentEdge1)
        else:
            break

    currentEdge1 = n
    # backward walk
    while True:
        next_edge = uniqueDict[currentEdge1][7]
        if type(next_edge) == int and next_edge not in outlist:
            currentEdge1 = next_edge
            outlist.add(currentEdge1)
        else:
            break

    lst = sorted(outlist)
    return lst

# ---------- Main class (keeps most original logic) ----------
class line():
    def __init__(self, inFile):
        self.name, self.ext = os.path.splitext(inFile)
        self.sf = shp.Reader(inFile)
        self.shape = self.sf.shapes()
        self.getProjection()
        self.getLines()

    def getProjection(self):
        prj_path = self.name + ".prj"
        if os.path.exists(prj_path):
            with open(prj_path, "r") as stream:
                self.projection = stream.read()
        else:
            self.projection = ""
        return(self.projection)

    def getLines(self):
        self.lines = [parts.points for parts in self.shape]

    def splitLines(self):
        outLine = []
        self.tempArray = []
        n = 0
        for line_pts in self.lines:
            for part in listToPairs(line_pts):
                outLine.append([part, computeOrientation(part), list(), list(), list(), list(), list(), list()])
                # use fixed 4 decimal string for endpoints as original
                self.tempArray.append([n, '%.4f_%.4f'%(part[0][0], part[0][1]), '%.4f_%.4f'%(part[1][0], part[1][1])])
                n += 1
        self.split = outLine

    def uniqueID(self):
        # create dict enumerating split lines
        self.unique = dict(enumerate(self.split))

    def getLinks(self):
        print("Finding adjacent segments...")
        # convert tempArray to numpy array of strings for faster masks
        # column 0 as ints, columns 1 & 2 strings
        temp_arr = np.array(self.tempArray, dtype=object)
        # ensure column 0 is integer dtype for efficiency
        temp_arr[:,0] = temp_arr[:,0].astype(np.int64)

        total = len(self.unique)
        indices = [(i, total) for i in range(total)]

        # set number of worker processes: leave 1 core free for UI
        n_workers = max(1, multiprocessing.cpu_count() - 1)
        # compute chunksize to balance IPC vs work
        chunksize = max(1, total // (n_workers * 4))

        # initialize pool once with temp array (sent once per worker)
        pool = multiprocessing.Pool(processes=n_workers, initializer=_init_worker, initargs=(temp_arr, None))
        try:
            result = pool.map(getLinksMultiprocessing_worker, indices, chunksize=chunksize)
        finally:
            pool.close()
            pool.join()

        # write back results into self.unique
        for a in result:
            n = a[0]
            self.unique[n][2] = a[1]
            self.unique[n][3] = a[2]

        print('>'*50 + ' [%d/%d] '%(len(self.unique),len(self.unique)) + '100%' + '\n', end='\r')

    def bestLink(self):
        self.anglePairs = dict()
        for edge in range(0,len(self.unique)):
            p1AngleSet = []
            p2AngleSet = []
            for link1 in self.unique[edge][2]:
                key = "%d_%d" % (edge, link1)
                self.anglePairs[key] = angleBetweenTwoLines(self.unique[edge][0], self.unique[link1][0])
                p1AngleSet.append(self.anglePairs[key])
            for link2 in self.unique[edge][3]:
                key = "%d_%d" % (edge, link2)
                self.anglePairs[key] = angleBetweenTwoLines(self.unique[edge][0], self.unique[link2][0])
                p2AngleSet.append(self.anglePairs[key])

            if p1AngleSet:
                val1, idx1 = max((val, idx) for (idx, val) in enumerate(p1AngleSet))
                self.unique[edge][4] = self.unique[edge][2][idx1], val1
            else:
                self.unique[edge][4] = 'DeadEnd'

            if p2AngleSet:
                val2, idx2 = max((val, idx) for (idx, val) in enumerate(p2AngleSet))
                self.unique[edge][5] = self.unique[edge][3][idx2], val2
            else:
                self.unique[edge][5] = 'DeadEnd'

    def crossCheckLinks(self, angleThreshold=0):
        print("Cross-checking and finalising the links...")
        L = len(self.unique)
        for edge in range(L):
            if edge%1000==0:
                currentProgress = math.floor(100*edge/L/2)
                remainingProgress = 50 - currentProgress
                print('>'*currentProgress + '-' * remainingProgress + ' [%d/%d] '%(edge,L) + '%d%%'%(currentProgress*2), end='\r')

            bestP1 = self.unique[edge][4][0]
            bestP2 = self.unique[edge][5][0]
            if type(bestP1) == type(1) and \
               edge in [self.unique[bestP1][4][0], self.unique[bestP1][5][0]] and \
               self.anglePairs["%d_%d" % (edge, bestP1)] > angleThreshold:
                self.unique[edge][6] = bestP1
            else:
                self.unique[edge][6] = 'LineBreak'

            if type(bestP2) == type(1) and \
               edge in [self.unique[bestP2][4][0], self.unique[bestP2][5][0]] and \
               self.anglePairs["%d_%d" % (edge, bestP2)] > angleThreshold:
                self.unique[edge][7] = bestP2
            else:
                self.unique[edge][7] = 'LineBreak'

        print('>'*50 + ' [%d/%d] '%(L,L) + '100%' + '\n', end='\r')

    def addLine(self, edge, parent=None, child='Undefined'):
        if child=='Undefined':
            self.mainEdge = len(self.merged)
        if not edge in self.assignedList:
            if parent==None:
                currentid = len(self.merged)
                self.merged[currentid] = set()
            else:
                currentid = self.mainEdge
            self.merged[currentid].add(listToTuple(self.unique[edge][0]))
            self.assignedList.append(edge)
            link1 = self.unique[edge][6]
            link2 = self.unique[edge][7]
            if type(1) == type(link1):
                self.addLine(link1, parent=edge, child=self.mainEdge)
            if type(1) == type(link2):
                self.addLine(link2, parent=edge, child=self.mainEdge)

    def mergeLines(self):
        print('Merging Lines...')
        self.mergingList = list()
        self.merged = list()

        total = len(self.unique)
        indices = [(i, total) for i in range(total)]

        # set number of worker processes
        n_workers = max(1, multiprocessing.cpu_count() - 1)
        chunksize = max(1, total // (n_workers * 4))

        # pass unique dict to worker initializer
        # Note: sending large unique dict to workers happens only once (initializer)
        pool = multiprocessing.Pool(processes=n_workers, initializer=_init_worker, initargs=(None, self.unique))
        try:
            result = pool.map(mergeLinesMultiprocessing_worker, indices, chunksize=chunksize)
        finally:
            pool.close()
            pool.join()

        for tempList in result:
            if tempList not in self.mergingList:
                self.mergingList.append(tempList)
                self.merged.append({listToTuple(self.unique[key][0]) for key in tempList})

        self.merged = dict(enumerate(self.merged))
        print('>'*50 + ' [%d/%d] '%(len(self.unique),len(self.unique)) + '100%' + '\n', end='\r')

    def exportPreMerge(self, outFile=None, unique = True):
        if outFile == None:
            outFile = "%s_%s_pythonScriptHierarchy.shp" % (time.strftime('%Y%m%d')[2:], self.name)
        with shp.Writer(outFile) as w:
            fields = ['UniqueID', 'Orientation', 'linksP1', 'linksP2', 'bestP1', 'bestP2', 'P1Final', 'P2Final']
            for f in fields:
                w.field(f, 'C')
            for parts in range(len(self.unique)):
                lineList = tupleToList(self.unique[parts][0])
                w.line([lineList])
                w.record(parts, self.unique[parts][1], self.unique[parts][2], self.unique[parts][3], self.unique[parts][4], self.unique[parts][5], self.unique[parts][6], self.unique[parts][7])
        self.setProjection(outFile)

    def exportStrokes(self, outFile=None):
        if outFile == None:
            outFile = "%s_%s_pythonScriptHierarchy.shp" % (time.strftime('%Y%m%d')[2:], self.name)
        with shp.Writer(outFile) as w:
            fields = ['ID', 'nSegments']
            for field in fields:
                w.field(field, 'C')
            for a in self.merged:
                w.record(a, len(self.merged[a]))
                linelist = tupleToList(list(self.merged[a]))
                w.line(linelist)
        self.setProjection(outFile)

    def setProjection(self, outFile):
        outName, ext = os.path.splitext(outFile)
        if hasattr(self, 'projection'):
            with open(outName + ".prj", "w") as stream:
                stream.write(self.projection)

# -------------------- Main execution --------------------
if __name__ == '__main__':
    # path to shapefile folder - set this to your folder in Colab (e.g., "/content/Chennai")
    myDir = r"C:\Users\Sonal.Ganvir\COINS\Data\Input\Delhi_final_roads_WGS_indicators"   # <<< update to your folder in Colab
    os.chdir(myDir)

    for file in glob.glob("*.shp"):
        t1 = time.time()
        print('Processing file..\n%s\n' % (file))
        myStreet = line(file)
        myStreet.splitLines()
        myStreet.uniqueID()
        myStreet.getLinks()
        myStreet.bestLink()
        myStreet.crossCheckLinks(angleThreshold=0)
        myStreet.mergeLines()
        myStreet.exportPreMerge(outFile=None)
        myStreet.exportStrokes(outFile=None)
        t2 = time.time()
        minutes = math.floor((t2-t1) / 60)
        seconds = (t2 - t1) % 60
        print("Processing complete in %d minutes %.2f seconds." % (minutes, seconds))
