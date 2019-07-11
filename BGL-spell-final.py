"""
Description : This file implements the Spell algorithm for log parsing
Author      : LogPAI team
License     : MIT
"""

import sys
import re
import os
import numpy as np
import pandas as pd
import hashlib
from datetime import datetime
import string
import time


class LCSObject:
    """ Class object to store a log group with the same template
    """
    def __init__(self, logTemplate='', logIDL=[]):
        self.logTemplate = logTemplate
        self.logIDL = logIDL


class Node:
    """ A node in prefix tree data structure
    """
    def __init__(self, token='', templateNo=0):
        self.logClust = None
        self.token = token
        self.templateNo = templateNo
        self.childD = dict()


class LogParser:
    """ LogParser class
    Attributes
    ----------
        path : the path of the input file
        logName : the file name of the input file
        savePath : the path of the output file
        tau : how much percentage of tokens matched to merge a log message
    """
    def __init__(self, indir='./', outdir='./result/', log_format=None, tau=0.5, rex=[], keep_para=True):
        self.path = indir
        self.logName = None
        self.savePath = outdir
        self.tau = tau
        self.logformat = log_format
        self.df_log = None
        self.rex = rex
        self.keep_para = keep_para
        self.lcsnum = 0
        self.prenum = 0
        self.simnum = 0
        self.lcstime = 0
        self.pretime = 0
        self.simtime = 0
        self.nenum = 0
        self.netime = 0

    def LCS(self, seq1, seq2):
        lengths = [[0 for j in range(len(seq2)+1)] for i in range(len(seq1)+1)]
        # row 0 and column 0 are initialized to 0 already
        for i in range(len(seq1)):
            for j in range(len(seq2)):
                if seq1[i] == seq2[j]:
                    lengths[i+1][j+1] = lengths[i][j] + 1
                else:
                    lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])

        # read the substring out from the matrix
        result = []
        lenOfSeq1, lenOfSeq2 = len(seq1), len(seq2)
        while lenOfSeq1!=0 and lenOfSeq2 != 0:
            if lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1-1][lenOfSeq2]:
                lenOfSeq1 -= 1
            elif lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1][lenOfSeq2-1]:
                lenOfSeq2 -= 1
            else:
                assert seq1[lenOfSeq1-1] == seq2[lenOfSeq2-1]
                result.insert(0,seq1[lenOfSeq1-1])
                lenOfSeq1 -= 1
                lenOfSeq2 -= 1
        return result

    def mergelog(self,logClustL):
        for logclust in logClustL:
            s =logclust.logTemplate
            for j in range(0,len(s)-1):
                if (j==(len(s)-2)) and (s[j]=='*') and ((s[j+1] == ',') or (s[j+1] == '=') or (s[j+1] == ':')):
                    del s[j+1]
                    break
                if (j==(len(s)-2)):
                    break
                if ((s[j]=='*') and (s[j+2] == ' ')and ((s[j+1] == ',') or (s[j+1] == '=') or (s[j+1] == ':'))):
                    del s[j+1]

    def seqlens1(self,seq):
        s = seq
        num1 = s.count('*')
        num2 = s.count(' ')
        return len(s) - num1 -num2

    def have_number(self,num):
        pattern = re.compile('[0-9]+')
        result = pattern.findall(num)
        if result:
            return True
        else:
            return False

    def seqlens2(self, seq):
        s = seq
        num1 = s.count('*')
        num2 = s.count(' ')
        num3 = 0
        num4 = 0
        for i in range(0,len(seq)):
            if(self.have_number(s[i])):
                num3 = num3 + 1
            if(s[i].isalpha()):
                num4 = num4 + 1
        return len(s) - num1 -num2 -num3 + num4

    def tokennum(self,seq1,seq2):
        num = 0
        c = 0
        for i in range(len(seq1)):
            for j in range(c, len(seq2)):
                if (seq1[i] == ' '):
                    break
                if (seq1[i] == seq2[j]):
                    c = j
                    num = num + 1
                    break
        return num

    def preciseseq(self,logClustL,result2):
        num2 = 0
        with open(os.path.join(self.savePath, result2), 'r') as fin:
            lines = fin.read().splitlines()
        num1 = len(lines)
        num3 = 0
        for i in range(0,num1):
          for logclust in logClustL:
                  template_str = ''.join(logclust.logTemplate)
                  if (lines[i] == template_str):
                     num2 = num2 + 1
                     num3 = num3 + len(logclust.logIDL)
                     break
        print('accuracy:',num2)
        print('accuracy:{0:.1f}%'.format(num3 * 100.0 / len(self.df_log)))

    def SimpleLoopMatch(self, logClustL, seq):
        for logClust in reversed(logClustL):
            if float(self.seqlens1(logClust.logTemplate)) < 0.5 * self.seqlens1(seq):
                continue
            # Check the template is a subsequence of seq (we use set checking as a proxy here for speedup since
            # incorrect-ordering bad cases rarely occur in logs)
            # token_set = set(seq)
            # if all(token in token_set or token == '*' for token in logClust.logTemplate):
            #     print('simpleLoopMatch:')
            #     print(seq)
            #     print(logClust.logTemplate)
            #     self.simnum = self.simnum + 1
            #     return logClust
            c = 0
            j = 0
            for i in range(len(logClust.logTemplate)):
                if (j == len(seq)):
                    j = 0
                    break
                if (logClust.logTemplate[i] == ' '):
                   j = j + 1
                   continue
                if (logClust.logTemplate[i] == '*'):
                   j = j + 1
                   if (i == len(logClust.logTemplate) - 1):
                       j=len(seq)
                   continue
                if (logClust.logTemplate[i] == seq[j]):
                   j = j + 1
                   continue
                break
            if (j == len(seq)):
                print('simpleLoopMatch:')
                print(seq)
                print(logClust.logTemplate)
                self.simnum = self.simnum + 1
                return logClust
        return None

    def PrefixTreeMatch(self, parentn, seq, idx):
        retLogClust = None
        length = self.seqlens1(seq)
        for i in range(idx, len(seq)):
            if seq[i] in parentn.childD:
                childn = parentn.childD[seq[i]]
                if (childn.logClust is not None):
                    constLM = [w for w in childn.logClust.logTemplate if w != '*']
                    if float(self.seqlens1(constLM)) >= self.tau * length:
                        return childn.logClust
                else:
                    return self.PrefixTreeMatch(childn, seq, i + 1)

        return retLogClust

    def LCSMatch(self, logClustL, seq):
        retLogClust = None

        maxLen = -1
        maxlcs = []
        maxClust = None
        # set_seq = set(seq)
        size_seq = self.seqlens2(seq)

        for logClust in reversed(logClustL):
            # set_template = set(logClust.logTemplate)
            #
            # if self.seqlens2(set_seq & set_template) < 0.5 * size_seq:
            #     continue
            size_lcs = self.tokennum(seq,logClust.logTemplate)

            if ((size_lcs < 0.5 * size_seq) or (size_lcs < 0.5 * self.seqlens2(logClust.logTemplate))):
                continue
            if ((len(logClust.logTemplate)) != len(seq)):
                 continue
            lcs = self.LCS(seq, logClust.logTemplate)
            if self.seqlens1(lcs) > maxLen or (self.seqlens1(lcs) == maxLen and self.seqlens1(logClust.logTemplate) < self.seqlens1(maxClust.logTemplate)):
                maxLen = self.seqlens1(lcs)
                maxlcs = lcs
                maxClust = logClust
                self.lcsnum = self.lcsnum + 1
                print('LCSMATCH:')
                print(seq)
                print(logClust.logTemplate)

        # LCS should be large then tau * len(itself)
        if float(maxLen) >= self.tau * size_seq:
            retLogClust = maxClust

        return retLogClust

    def getTemplate(self, lcs, seq):
        retVal = []
        if not lcs:
            return retVal

        lcs = lcs[::-1]
        i = 0
        for token in seq:
            i += 1
            if token == lcs[-1]:
                retVal.append(token)
                lcs.pop()
            else:
                retVal.append('*')
            if not lcs:
                break
        if i < len(seq):
            retVal.append('*')
        return retVal

    def addSeqToPrefixTree(self, rootn, newCluster):
        parentn = rootn
        seq = newCluster.logTemplate
        seq = [w for w in seq if w != '*']

        for i in range(len(seq)):
            tokenInSeq = seq[i]
            # Match
            if tokenInSeq in parentn.childD:
                parentn.childD[tokenInSeq].templateNo += 1
            # Do not Match
            else:
                parentn.childD[tokenInSeq] = Node(token=tokenInSeq, templateNo=1)
            parentn = parentn.childD[tokenInSeq]

        if parentn.logClust is None:
            parentn.logClust = newCluster

    def removeSeqFromPrefixTree(self, rootn, newCluster):
        parentn = rootn
        seq = newCluster.logTemplate
        seq = [w for w in seq if w != '*']

        for tokenInSeq in seq:
            if tokenInSeq in parentn.childD:
                matchedNode = parentn.childD[tokenInSeq]
                if matchedNode.templateNo == 1:
                    del parentn.childD[tokenInSeq]
                    break
                else:
                    matchedNode.templateNo -= 1
                    parentn = matchedNode

    def outputResult(self, logClustL):

        templates = [0] * self.df_log.shape[0]
        ids = [0] * self.df_log.shape[0]
        df_event = []

        for logclust in logClustL:
            template_str = ''.join(logclust.logTemplate)
            eid = hashlib.md5(template_str.encode('utf-8')).hexdigest()[0:8]
            for logid in logclust.logIDL:
                templates[logid - 1] = template_str
                ids[logid - 1] = eid
            df_event.append([eid, template_str, len(logclust.logIDL)])

        df_event = pd.DataFrame(df_event, columns=['EventId', 'EventTemplate', 'Occurrences'])

        self.df_log['EventId'] = ids
        self.df_log['EventTemplate'] = templates
        if self.keep_para:
            self.df_log["ParameterList"] = self.df_log.apply(self.get_parameter_list, axis=1)
        self.df_log.to_csv(os.path.join(self.savePath, self.logname + '_structured.csv'), index=False)
        df_event.to_csv(os.path.join(self.savePath, self.logname + '_templates.csv'), index=False)

    def printTree(self, node, dep):
        pStr = ''
        for i in range(dep):
            pStr += '\t'

        if node.token == '':
            pStr += 'Root'
        else:
            pStr += node.token
            if node.logClust is not None:
                pStr += '-->' + ''.join(node.logClust.logTemplate)
        print(pStr + ' ('+ str(node.templateNo) + ')')

        for child in node.childD:
            self.printTree(node.childD[child], dep + 1)

    def parse(self, logname):
        starttime = datetime.now()
        print('Parsing file: ' + os.path.join(self.path, logname))
        self.logname = logname
        self.load_data()
        rootNode = Node()
        logCluL = []

        count = 0
        for idx, line in self.df_log.iterrows():
            logID = line['LineId']
            logmessageL = list(filter(lambda x: x != '', re.split(r'([\s:])', self.preprocess(line['Content']))))
            constLogMessL = [w for w in logmessageL if w != '*']

            # Find an existing matched log cluster
            if (self.seqlens1(logmessageL) <= 0):
                continue
            starttime1 = time.clock()
            matchCluster = self.PrefixTreeMatch(rootNode, constLogMessL, 0)
            endtime1 = time.clock()
            self.pretime = self.pretime + endtime1 - starttime1
            self.prenum = self.prenum + 1

            if matchCluster is None:
                self.prenum = self.prenum - 1
                starttime2 = time.clock()
                matchCluster = self.SimpleLoopMatch(logCluL, constLogMessL)
                endtime2 = time.clock()
                self.simtime = self.simtime + endtime2 - starttime2

                if matchCluster is None:
                    starttime3 = time.clock()
                    matchCluster = self.LCSMatch(logCluL, logmessageL)
                    endtime3 = time.clock()
                    self.lcstime = self.lcstime + endtime3 - starttime3

                    # Match no existing log cluster
                    if matchCluster is None:
                        self.nenum = self.nenum + 1
                        starttime4 = time.clock()
                        print('newlog:',logmessageL)
                        newCluster = LCSObject(logTemplate=logmessageL, logIDL=[logID])
                        logCluL.append(newCluster)
                        self.addSeqToPrefixTree(rootNode, newCluster)
                        endtime4 = time.clock()
                        self.netime = self.netime + endtime4 - starttime4
                    # Add the new log message to the existing cluster
                    else:
                        newTemplate = self.getTemplate(self.LCS(logmessageL, matchCluster.logTemplate),
                                                       matchCluster.logTemplate)
                        if ''.join(newTemplate) != ''.join(matchCluster.logTemplate):
                            self.removeSeqFromPrefixTree(rootNode, matchCluster)
                            matchCluster.logTemplate = newTemplate
                            self.addSeqToPrefixTree(rootNode, matchCluster)
            if matchCluster:
                matchCluster.logIDL.append(logID)
            count += 1
            if count % 1000 == 0 or count == len(self.df_log):
                print('prenum:', self.prenum)
                print('pretime:[{!s}]'.format(self.pretime))
                print('simnum:', self.simnum)
                print('simtime:[{!s}]'.format(self.simtime))
                print('lcsnum:', self.lcsnum)
                print('lcstime:[{!s}]'.format(self.lcstime))
                print('nenum:', self.nenum)
                print('netime:[{!s}]'.format(self.netime))
                print('Processed {0:.1f}% of log lines.'.format(count * 100.0 / len(self.df_log)))

        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)

        self.mergelog(logCluL)
        self.outputResult(logCluL)
        self.printTree(rootNode, 0)
        print('prenum:', self.prenum)
        print('pretime:{!s}'.format(self.pretime))
        print('simnum:', self.simnum)
        print('simtime:{!s}'.format(self.simtime))
        print('lcsnum:', self.lcsnum)
        print('lcstime:{!s}'.format(self.lcstime))
        print('nenum:', self.nenum)
        print('netime:{!s}'.format(self.netime))
        self.preciseseq(logCluL,'result1.log')
        print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - starttime))

    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.logformat)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logname), regex, headers, self.logformat)

    def preprocess(self, line):
        for currentRex in self.rex:
            line = re.sub(currentRex, '*', line)
        return line

    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """ Function to transform log file to dataframe
        """
        log_messages = []
        linecount = 0
        with open(log_file, 'r') as fin:
            for line in fin.readlines():
                line = re.sub(r'[^\x00-\x7F]+', '<NASCII>', line)
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    pass
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        return logdf

    def generate_logformat_regex(self, logformat):
        """ Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        print(splitters)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex

    def get_parameter_list(self, row):
        template_regex = re.sub(r"\s<.{1,5}>\s", "*", row["EventTemplate"])
        if "*" not in template_regex: return []
        template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template_regex)
        print(31)
        template_regex = re.sub(r'\\ +', r'[^A-Za-z0-9]+', template_regex)
        print(32)
        template_regex = "^" + template_regex.replace("\*", "(.*?)") + "$"
        print(33)
        print(row["EventTemplate"])
        print(template_regex)
        parameter_list = re.findall(template_regex, row["Content"])
        print(34)
        parameter_list = parameter_list[0] if parameter_list else ()
        print(35)
        parameter_list = list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list]
        print(36)
        parameter_list = [para.strip(string.punctuation).strip(' ') for para in parameter_list]
        return parameter_list
t=LogParser('./logs-collection','./result/','<C1> <Num1> <Day> <C2> <time> <C3> <C4> <C5> <Level> <Content>',0.5,[],False)
t.parse('BGL.log')