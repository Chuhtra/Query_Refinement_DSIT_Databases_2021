class LibraryOfTime:
    """
    Class for the timekeeper object. In this object's attributes the time durations (used for the experimental part)
    will be organized.
    """

    def __init__(self):
        self.timeToConnectDB = 0
        self.timeToReadQuery = 0
        self.timeToValidateSyntax = 0
        self.timeToExecuteQuery = 0

        self.timeToFixDictionary = 0

        self.timeToGetActualData = 0
        self.timeToGetUDFMapReady = 0

        self.timeToCreateBST = 0
        self.timeToExploreBST = 0
        self.timeToCheckConstraints = 0
        self.timeToPrepFails = 0

        self.timeToRunRelaxation = 0
        self.timeToRankAtClient = 0

        self.udf_size = 0  # THIS WAS ONLY USED FOR THE SCALABILITY EXPERIMENT RUN

    def getExecutionTime(self):
        """
        Simple method that returns the part of the execution that may vary based on query parameters.

        :return: Total query execution time, from data access to finish.
        """

        return self.timeToGetActualData + self.timeToGetUDFMapReady + self.timeToCreateBST + self.timeToExploreBST + \
               self.timeToCheckConstraints + self.timeToPrepFails + self.timeToRunRelaxation + self.timeToRankAtClient

    def getInitialTime(self):
        """
        Simple method that returns the part of the execution that is irrelevant to the query parameters.

        :return: Running time from db connection to query dict. preparation.
        """

        return self.timeToConnectDB + self.timeToReadQuery + self.timeToValidateSyntax + self.timeToFixDictionary

    def purge(self):
        """
        This function just returns a timekeeper object to its initialization, only useful for Scalability experiment,
        due to repeated querying with same timekeeper object.
        """
        self.timeToConnectDB = 0
        self.timeToReadQuery = 0
        self.timeToValidateSyntax = 0
        self.timeToExecuteQuery = 0

        self.timeToFixDictionary = 0

        self.timeToGetActualData = 0
        self.timeToGetUDFMapReady = 0

        self.timeToCreateBST = 0
        self.timeToExploreBST = 0
        self.timeToCheckConstraints = 0
        self.timeToPrepFails = 0

        self.timeToRunRelaxation = 0
        self.timeToRankAtClient = 0

        self.udf_size = 0


def saveTimers(obj, filename):
    """
    Simple function that store the object to be available later for analysis.

    :param obj: Object containing recorded execution durations as attributes.
    :param filename: The selected filename for the stored file.
    """
    import csv
    with open(filename, 'w', ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Name', 'Seconds'])

        attributes = [y for y in dir(obj) if not y.startswith('_')]
        for duration in attributes:
            writer.writerow([duration, getattr(obj, duration)])


timekeeper = LibraryOfTime()


def startTime():
    """
    Initialization method that sets a timekeeper object to be available globally.
    """
    global timekeeper
