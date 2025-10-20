import numpy
import argparse
from utils import config_utils, filepath_utils
import os
import csv


def average_results(filePath, stain, label, maxiter, classdefinitions):

    predictionorder = []
    for className in classdefinitions.keys():
        predictionorder.append(classdefinitions[className][0])
    indexes = numpy.argsort(predictionorder)
    classNames = [list(classdefinitions)[i] for i in indexes]

    results = {}
    with open(os.path.join(filePath.get_result_path(label + "_" + str(1)), "results_maxoutput_%s.%s_%d.txt" % (stain, label, 1)), 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        firstrow = True
        for row in reader:
            if not firstrow and len(row) == 5:
                imagename = row[0]
                results[imagename] = {}
                results[imagename]['overall'] = []
                for classname in classNames:
                    results[imagename][classname] = []
            #elif not firstrow and len(row) == 6:
            #    results[imagename][row[1]] = []
            else:
                firstrow = False

    for i in range(1, maxiter+1):
        with open(os.path.join(filePath.get_result_path(label + "_" + str(i)), "results_maxoutput_%s.%s_%d.txt" % (stain, label, i)), 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            firstrow = True
            for row in reader:
                if not firstrow and len(row) == 5:
                    imagename = row[0]
                    results[imagename]['overall'].append([float(a) for a in row[-4:]])
                elif not firstrow and len(row) == 6:
                    results[imagename][row[1]].append([float(a) for a in row[-4:]])
                else:
                    firstrow = False

    averages = {}
    stddevs = {}
    for imagekey in results:
        averages[imagekey] = {}
        stddevs[imagekey] = {}

    for imagekey in results:
        for classname in classNames:
            resultsarray = numpy.array(results[imagekey][classname])
            averages[imagekey][classname] = numpy.average(resultsarray, axis=0)
            stddevs[imagekey][classname] = numpy.std(resultsarray, axis=0)
        averages[imagekey]['overall'] = numpy.average(resultsarray, axis=0)
        stddevs[imagekey]['overall'] = numpy.std(resultsarray, axis=0)

    if not os.path.exists(filePath.get_average_result_path(label)):
        os.makedirs(filePath.get_average_result_path(label))

    with open(os.path.join(filePath.get_average_result_path(label), "results_maxoutput_average_%s.%s.txt" % (stain, label)), 'w') as f:
        writer = csv.writer(f, delimiter='\t')

        writer.writerow(numpy.array(('patientNb', 'F1', 'precision', 'recall', 'accuracy')))

        for imagekey, imageclasses in sorted(results.items()):
            if imagekey != 'overall':
                writer.writerow(['%s' % imagekey] + ['%.3f' % a for a in averages[imagekey]['overall']])
                for classname in classNames:
                    writer.writerow(['%s' % classname] + ['%.3f' % a for a in averages[imagekey][classname]])
        writer.writerow(['overall'] + ['%.3f' % a for a in averages['overall']['overall']])
        for classname in classNames:
            writer.writerow(['%s' % classname] + ['%.3f' % a for a in averages['overall'][classname]])

    with open(os.path.join(filePath.get_average_result_path(label), "results_maxoutput_stddev_%s.%s.txt" % (stain, label)), 'w') as f:
        writer = csv.writer(f, delimiter='\t')

        writer.writerow(numpy.array(('patientNb', 'F1', 'precision', 'recall', 'accuracy')))

        for imagekey, imageclasses in sorted(results.items()):
            if imagekey != 'overall':
                writer.writerow(['%s' % imagekey] + ['%.3f' % a for a in stddevs[imagekey]['overall']])
                for classname in classNames:
                    writer.writerow(['%s' % classname] + ['%.3f' % a for a in stddevs[imagekey][classname]])
        writer.writerow(['overall'] + ['%.3f' % a for a in stddevs['overall']['overall']])
        for classname in classNames:
            writer.writerow(['%s' % classname] + ['%.3f' % a for a in stddevs['overall'][classname]])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Average results.')

    parser.add_argument('label', type=str, help='the label of the model to be averaged (without postfix number)')
    parser.add_argument('maxit', type=str, help='maximum iteration to average, i.e. 1, ..., maxit')

    parser.add_argument('-c', '--configfile', type=str, help='the configuration file to use')
    parser.add_argument('-s', '--teststain', type=str, help='the stain to test upon')
    parser.add_argument('-m', '--maxoutput', action='store_const', default=False, const=True,
                        help='test maxoutput')
    parser.add_argument('-t', '--threshold', action='store_const', default=False, const=True,
                        help='test threshold levels')
    args = parser.parse_args()

    if not args.maxoutput and not args.threshold:
        args.maxoutput = True
        args.threshold = True

    if args.configfile:
        config = config_utils.readconfig(args.configfile)
    else:
        config = config_utils.readconfig()

    if args.label:
        label = args.label
    else:
        configfilename = glob.glob(os.path.join(config['detector.outputpath'], 'sysmifta.*.cfg'))
        if len(configfilename) > 1:
            raise RuntimeError('Cannot infer the label when more than one config file exists')
        if len(configfilename) == 0:
            raise RuntimeError('No config file exists')
        label = os.path.splitext(os.path.basename(configfilename[0]))[0].split(".")[1]

    config = config_utils.readconfig(os.path.join(config['detector.outputpath'], 'sysmifta.' + label + '_1.cfg'))

    filePath = filepath_utils.FilepathGenerator(config)

    if args.teststain:
        teststain = args.teststain
    else:
        teststain = '*'

    # Evaluate maximum classifier output
    average_results(filePath, teststain, label, int(args.maxit), config['extraction.class_definitions'])
