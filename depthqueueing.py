from collections import deque


class DepthQueueModifier:

    def __init__(self, maxlen=5, breakfree=0.1):
        # queue consists of [object depth, boundingbox]
        self.queue = deque(maxlen=maxlen)
        self.breakfree = breakfree
        self.currentmean = None
        self.queue.extend([[1,1]])

    def update_calc_average(self):
        if not self.queue:
            self.currentmean = None
        else:
            # Extract depths from queue
            depths = [item[0] for item in self.queue]
            self.currentmean = sum(depths) / len(depths)

    def add_item_to_queue(self, objectdepth, boundingbox):
        self.queue.appendleft([objectdepth, boundingbox])

    def return_currentmean(self):
        return self.currentmean

    def check_breakfree(self):
        threshold = self.currentmean * self.breakfree
        if self.currentmean - threshold > newvalue or newvalue > self.currentmean + threshold:
            return True
        else:
            return False

    '''
    def run(self):
        self.update_calc_average()


        self.queue.appendleft([newvalue])
        self.currentmean = DepthQueueModifier.calc_average(self.queue)

        print(f"after:{self.queue}")


   

    def run(self, newvalue):
        if self.queue:
            print(f"before:{self.queue}")
            print(f"mean:{DepthQueueModifier.calc_average(self.queue)}")
            self.currentmean = DepthQueueModifier.calc_average(self.queue)

            threshold = self.currentmean * self.breakfree
            if self.currentmean - threshold > newvalue or newvalue > self.currentmean + threshold:
                self.queue.appendleft([newvalue])
                self.currentmean = DepthQueueModifier.calc_average(self.queue)
                print("triggered")
                print(f"currentmean:{self.currentmean}")
                print(f"current queue {self.queue}")
                return self.currentmean
            else:
                return self.currentmean
        else:
            self.queue.append(newvalue)

        print(f"after:{self.queue}")
    '''

class QueueManager:

    def __init__(self, IoU=0.3):
        self.suitable_object_found = False
        self.IoU = IoU
        self.currentmeandepth = None

    def check_objects_and_return_depth(self, objectdepth, roi, objectlist):
        self.suitable_object_found = False
        for object in objectlist:
            if QueueManager.bb_intersection_over_union(roi, object.queue[0][1]) > self.IoU:
                object.queue.appendleft([objectdepth, roi])
                object.update_calc_average()
                self.suitable_object_found = True
                print('triggered')
                return object.return_currentmean()




        if self.suitable_object_found == False:
            newobject = DepthQueueModifier()
            newobject.add_item_to_queue(objectdepth, roi)
            objectlist.append(newobject)
            return objectdepth
        #else:
            #return objectdepth


    def bb_intersection_over_union(boxA, boxB):
        # Example bounding box coordinates: (x1, y1, x2, y2)
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection area
        # and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

        


if __name__ == '__main__':
    d = DepthQueueModifier()
    print(d.run(4))
    print(d.run(5))
    print(d.run(6))
