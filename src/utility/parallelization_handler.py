from typing import Callable, Any
from multiprocessing import Process, cpu_count
from numpy import ndarray, array, append


class Parallelize(object):

    @staticmethod
    def parallelizeOneWithOne(action: Callable, *args: tuple[Any, ...]) -> Process:
        newProcess: Process = Process(target=action, args=args)
        newProcess.start()
        return newProcess

    @staticmethod
    def parallelizeOneWithMany(action: Callable, argsCollection: ndarray[tuple[Any, ...]]) -> ndarray[Process]:
        runningProcesses: ndarray[Process] = array([])

        for args in argsCollection:
            newProcess: Process = Parallelize.parallelizeOneWithOne(action, *args)
            runningProcesses = append(runningProcesses, [newProcess])

        return runningProcesses

    @staticmethod
    def waitProcesses(processes: ndarray[Process], action: Callable, args: tuple[Any, ...]):
        for idx in range(processes.size):
            processes[idx].join()

        action(*args)

    @staticmethod
    def getMaxProcessesNumber() -> int:
        return cpu_count()

