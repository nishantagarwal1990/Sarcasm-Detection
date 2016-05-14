#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple Python wrapper for runTagger.sh script for CMU's Tweet Tokeniser and Part of Speech tagger: http://www.ark.cs.cmu.edu/TweetNLP/
Usage:
results=runtagger_parse(['example tweet 1', 'example tweet 2'])
results will contain a list of lists (one per tweet) of triples, each triple represents (term, type, confidence)
"""
import subprocess
import shlex

# The only relavent source I've found is here:
# http://m1ked.com/post/12304626776/pos-tagger-for-twitter-successfully-implemented-in
# which is a very simple implementation, my implementation is a bit more
# useful (but not much).

# NOTE this command is directly lifted from runTagger.sh
RUN_TAGGER_CMD = "java -XX:ParallelGCThreads=2 -Xmx500m -jar ark-tweet-nlp-0.3.2.jar"


def _split_results(rows):
    """Parse the tab-delimited returned lines, modified from: https://github.com/brendano/ark-tweet-nlp/blob/master/scripts/show.py"""
    for line in rows:
        line = line.strip()  # remove '\n'
        if len(line) > 0:
            if line.count('\t') == 2:
                parts = line.split('\t')
                tokens = parts[0]
                tags = parts[1]
                confidence = float(parts[2])
                yield tokens, tags, confidence

def _split_results_for_tags(rows):
    """Parse the tab-delimited returned lines, modified from: https://github.com/brendano/ark-tweet-nlp/blob/master/scripts/show.py"""
    for line in rows:
        line = line.strip()  # remove '\n'
        if len(line) > 0:
            if line.count('\t') == 2:
                parts = line.split('\t')
                #tokens = parts[0]
                tags = parts[1]
                #confidence = float(parts[2])
                #yield tokens, tags, confidence
                yield tags

def _split_results_for_tokens(rows):
    """Parse the tab-delimited returned lines, modified from: https://github.com/brendano/ark-tweet-nlp/blob/master/scripts/show.py"""
    for line in rows:
        line = line.strip()  # remove '\n'
        if len(line) > 0:
            if line.count('\t') == 2:
                parts = line.split('\t')
                tokens = parts[0]
                #tags = parts[1]
                #confidence = float(parts[2])
                #yield tokens, tags, confidence
                yield tokens


def _call_runtagger(tweets, run_tagger_cmd=RUN_TAGGER_CMD):
    """Call runTagger.sh using a named input file"""

    # remove carriage returns as they are tweet separators for the stdin
    # interface
    tweets_cleaned = [tw.replace('\n', ' ') for tw in tweets]
    message = "\n".join(tweets_cleaned)

    # force UTF-8 encoding (from internal unicode type) to avoid .communicate encoding error as per:
    # http://stackoverflow.com/questions/3040101/python-encoding-for-pipe-communicate
    message = message.encode('utf-8')

    # build a list of args
    args = shlex.split(run_tagger_cmd)
    args.append('--output-format')
    args.append('conll')
    po = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # old call - made a direct call to runTagger.sh (not Windows friendly)
    #po = subprocess.Popen([run_tagger_cmd, '--output-format', 'conll'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result = po.communicate(message)
    # expect a tuple of 2 items like:
    # ('hello\t!\t0.9858\nthere\tR\t0.4168\n\n',
    # 'Listening on stdin for input.  (-h for help)\nDetected text input format\nTokenized and tagged 1 tweets (2 tokens) in 7.5 seconds: 0.1 tweets/sec, 0.3 tokens/sec\n')

    pos_result = result[0].strip('\n\n')  # get first line, remove final double carriage return
    pos_result = pos_result.split('\n\n')  # split messages by double carriage returns
    pos_results = [pr.split('\n') for pr in pos_result]  # split parts of message by each carriage return
    return pos_results


def runtagger_parse(tweets, run_tagger_cmd=RUN_TAGGER_CMD):
    """Call runTagger.sh on a list of tweets, parse the result, return lists of tuples of (term, type, confidence)"""
    pos_raw_results = _call_runtagger(tweets, run_tagger_cmd)
    pos_result = []
    tokens_results = []
    for pos_raw_result in pos_raw_results:
        pos_result.append([x for x in _split_results_for_tags(pos_raw_result)])
        tokens_results.append([x for x in _split_results_for_tokens(pos_raw_result)])
    return pos_result,tokens_results


def check_script_is_present(run_tagger_cmd=RUN_TAGGER_CMD):
    """Simple test to make sure we can see the script"""
    success = False
    try:
        args = shlex.split(run_tagger_cmd)
        args.append("--help")
        po = subprocess.Popen(args, stdout=subprocess.PIPE)
        # old call - made a direct call to runTagger.sh (not Windows friendly)
        #po = subprocess.Popen([run_tagger_cmd, '--help'], stdout=subprocess.PIPE)
        while not po.poll():
            lines = [l for l in po.stdout]
        # we expected the first line of --help to look like the following:
        assert "RunTagger [options]" in lines[0]
        success = True
    except OSError as err:
        print "Caught an OSError, have you specified the correct path to runTagger.sh? We are using \"%s\". Exception: %r" % (run_tagger_cmd, repr(err))
    return success


if __name__ == "__main__":
    #print "Checking that we can see \"%s\", this will crash if we can't" % (RUN_TAGGER_CMD)
    #success = check_script_is_present()
    #if success:
        #print "Success."
        #print "Now pass in two messages, get a list of tuples back:"
        #tweets = ['this is a message', 'and a second message']
        #print runtagger_parse(tweets)
    inputfile = open("tagged_tweets.txt","r")
    #outputf = open("test.txt","w")

    tagged_tweets = list()
    cleaned_tweets = list()
    cleaned_tweets_sarcasm = list()
    cleaned_tweets_not_sarcasm = list()
    tagged_sarcastic_tweets = list()
    tagged_not_sarcastic_tweets = list()
    for line in inputfile.readlines():
        line = line.split()
        #print line
        #tweet = ""
        #for word in line:
        if "#sarcasm" in line:
            line.remove("#sarcasm")
        if "#Sarcasm" in line:
            line.remove("#Sarcasm")
        if "#Sarcasm." in line:
            line.remove("#Sarcasm.")
        if "#sarcasm." in line:
            line.remove("#sarcasm.")
        if "#SARCASM" in line:
            line.remove("#SARCASM")
        if "#SARCASM^" in line:
            line.remove("#SARCASM^")
        if "<-------#sarcasm" in line:
            line.remove("<-------#sarcasm")
        if "#sarcasm!" in line:
            line.remove("#sarcasm!")

        tweet = ""
        for i in xrange(1,len(line)):
            tweet += line[i]+ " "
            
        tweet = tweet.strip()

        if line[0] == "SARCASM":
            cleaned_tweets_sarcasm.append(tweet)
        else:
            cleaned_tweets_not_sarcasm.append(tweet)
            #if '#' not in word:
              #  tweet += word + " "
        #tweet = tweet.strip()
        #outputf.write(tweet+"\n")
        #cleaned_tweets.append(tweet)
    
    inputfile.close()
    #outputf.close()
    #print cleaned_tweets_sarcasm
    #print len(cleaned_tweets)
    #tagged_tweets = runtagger_parse(cleaned_tweets)
    tagged_sarcastic_tweets,tokens_sarcastic_tweets = runtagger_parse(cleaned_tweets_sarcasm)
    tagged_not_sarcastic_tweets,tokens_not_sarcastic_tweets = runtagger_parse(cleaned_tweets_not_sarcasm)
    #print len(cleaned_tweets_sarcasm)
    #print len(tagged_sarcastic_tweets)
    ngram_tag_list = list()
    sarcastic_bigram_tag_list = list()
    sarcastic_trigram_tag_list = list()
    not_sarcastic_bigram_tag_list = list()
    not_sarcastic_trigram_tag_list = list()

    sarcastic_features_bigram = dict()
    not_sarcastic_features_bigram = dict()
    sarcastic_features_trigram = dict()
    not_sarcastic_features_trigram = dict()
    sarcastic_features_quadgram = dict()
    not_sarcastic_features_quadgram = dict()
    sarcastic_features_pentgram = dict()
    not_sarcastic_features_pentgram = dict()
    sarcastic_bigram = 0
    sarcastic_trigram = 0
    not_sarcastic_bigram = 0
    not_sarcastic_trigram = 0
    sarcastic_quadgram = 0
    sarcastic_pentgram = 0
    not_sarcastic_quadgram = 0
    not_sarcastic_pentgram = 0


    print "Getting POS tag list for sarcastic tweets"
    for tag_tweet in tagged_sarcastic_tweets:
        #bigrams
        bigram = ""
        i = 0
        j = 1
        while j < len(tag_tweet):
            bigram = tag_tweet[i] + " " + tag_tweet[j]
            sarcastic_bigram += 1
            if bigram not in sarcastic_features_bigram.keys():
                sarcastic_features_bigram[bigram] = 1
            else:
                sarcastic_features_bigram[bigram] += 1
            i = i+1
            j = j+1
    
    
    for tag_tweet in tagged_sarcastic_tweets:
        #trigrams
        trigram = ""
        i = 0
        j = 1
        k = 2
        while k < len(tag_tweet):
            trigram = tag_tweet[i] + " " + tag_tweet[j] + " " + tag_tweet[k]
            sarcastic_trigram += 1
            if trigram not in sarcastic_features_trigram.keys():
                sarcastic_features_trigram[trigram] = 1
            else:
                sarcastic_features_trigram[trigram] += 1
            i += 1
            j += 1
            k += 1

    for tag_tweet in tagged_sarcastic_tweets:
        #trigrams
        quadgram = ""
        i = 0
        j = 1
        k = 2
        l = 3
        while l < len(tag_tweet):
            quadgram = tag_tweet[i] + " " + tag_tweet[j] + " " + tag_tweet[k] + " " + tag_tweet[l]
            sarcastic_quadgram += 1
            if quadgram not in sarcastic_features_quadgram.keys():
                sarcastic_features_quadgram[quadgram] = 1
            else:
                sarcastic_features_quadgram[quadgram] += 1
            i += 1
            j += 1
            k += 1
            l += 1

    for tag_tweet in tagged_sarcastic_tweets:
        #trigrams
        pentgram = ""
        i = 0
        j = 1
        k = 2
        l = 3
        m = 4
        while m < len(tag_tweet):
            pentgram = tag_tweet[i] + " " + tag_tweet[j] + " " + tag_tweet[k] + " " + tag_tweet[l] + " " + tag_tweet[m]
            sarcastic_pentgram += 1
            if pentgram not in sarcastic_features_pentgram.keys():
                sarcastic_features_pentgram[pentgram] = 1
            else:
                sarcastic_features_pentgram[pentgram] += 1
            i += 1
            j += 1
            k += 1
            l += 1
            m += 1


    print "Getting POS tag list for non sarcastic tweets"
    for tag_tweet in tagged_not_sarcastic_tweets:
        #bigrams
        bigram = ""
        i = 0
        j = 1
        while j < len(tag_tweet):
            bigram = tag_tweet[i] + " " + tag_tweet[j]
            not_sarcastic_bigram += 1
            if bigram not in not_sarcastic_features_bigram.keys():
                not_sarcastic_features_bigram[bigram] = 1
            else:
                not_sarcastic_features_bigram[bigram] += 1
            i = i + 1
            j = j + 1
    
    
    for tag_tweet in tagged_not_sarcastic_tweets:
        #trigrams
        trigram = ""
        i = 0
        j = 1
        k = 2
        while k < len(tag_tweet):
            trigram = tag_tweet[i] + " " + tag_tweet[j] + " " + tag_tweet[k]
            not_sarcastic_trigram += 1
            if trigram not in not_sarcastic_features_trigram.keys():
                not_sarcastic_features_trigram[trigram] = 1
            else:
                not_sarcastic_features_trigram[trigram] += 1
            i += 1
            j += 1
            k += 1

    for tag_tweet in tagged_not_sarcastic_tweets:
        #trigrams
        quadgram = ""
        i = 0
        j = 1
        k = 2
        l = 3
        while l < len(tag_tweet):
            quadgram = tag_tweet[i] + " " + tag_tweet[j] + " " + tag_tweet[k] + " " + tag_tweet[l]
            not_sarcastic_quadgram += 1
            if quadgram not in not_sarcastic_features_quadgram.keys():
                not_sarcastic_features_quadgram[quadgram] = 1
            else:
                not_sarcastic_features_quadgram[quadgram] += 1
            i += 1
            j += 1
            k += 1
            l += 1

    for tag_tweet in tagged_not_sarcastic_tweets:
        #trigrams
        pentgram = ""
        i = 0
        j = 1
        k = 2
        l = 3
        m = 4
        while m < len(tag_tweet):
            pentgram = tag_tweet[i] + " " + tag_tweet[j] + " " + tag_tweet[k] + " " + tag_tweet[l] + " " + tag_tweet[m]
            not_sarcastic_pentgram += 1
            if pentgram not in not_sarcastic_features_pentgram.keys():
                not_sarcastic_features_pentgram[pentgram] = 1
            else:
                not_sarcastic_features_pentgram[pentgram] += 1
            i += 1
            j += 1
            k += 1
            l += 1
            m += 1

    #print ngram_tag_list
    #feature_dict = dict()
    

    outputfile = open("prob_ngrams_cleaned.txt","w")

    bigram_list = list()
    print "Writing to file"
    outputfile.write("POS and Sarcasm Bigram Probability \n")
    for key in sarcastic_features_bigram:
        if key in not_sarcastic_features_bigram.keys():
            sar_bigram = sarcastic_features_bigram[key]/(float)(sarcastic_bigram)
            not_sar_bigram = not_sarcastic_features_bigram[key]/(float)(not_sarcastic_bigram)
            val = sar_bigram/(float)(sar_bigram+not_sar_bigram)
        else:
            val = 1
        if val >= 0.76:
            bigram_list.append(key)
        outputfile.write(str(key)+"\t"+str(val)+"\n")

    trigram_list = list()
    outputfile.write("\n\nPOS and Sarcasm Trigram Probability \n")
    for key in sarcastic_features_trigram:
        if key in not_sarcastic_features_trigram.keys():
            sar_trigram = sarcastic_features_trigram[key]/(float)(sarcastic_trigram)
            not_sar_trigram = not_sarcastic_features_trigram[key]/(float)(not_sarcastic_trigram)
            val = sar_trigram/(float)(sar_trigram+not_sar_trigram)
        else:
            val = 1
        if val >= 0.8:
            trigram_list.append(key)
        outputfile.write(str(key)+"\t"+str(val)+"\n")

    quadgram_list = list()
    outputfile.write("\n\nPOS and Sarcasm Quadgram Probability \n")
    for key in sarcastic_features_quadgram:
        if key in not_sarcastic_features_quadgram.keys():
            sar_quadgram = sarcastic_features_quadgram[key]/(float)(sarcastic_quadgram)
            not_sar_quadgram = not_sarcastic_features_quadgram[key]/(float)(not_sarcastic_quadgram)
            val = sar_quadgram/(float)(sar_quadgram+not_sar_quadgram)
        else:
            val = 1
        if val >= 0.8:
            quadgram_list.append(key)
        outputfile.write(str(key)+"\t"+str(val)+"\n")

    pentgram_list = list()
    outputfile.write("\n\nPOS and Sarcasm Pentgram Probability \n")
    for key in sarcastic_features_pentgram:
        if key in not_sarcastic_features_pentgram.keys():
            sar_pentgram = sarcastic_features_pentgram[key]/(float)(sarcastic_pentgram)
            not_sar_pentgram = not_sarcastic_features_pentgram[key]/(float)(not_sarcastic_pentgram)
            val = sar_pentgram/(float)(sar_pentgram+not_sar_pentgram)
        else:
            val = 1
        if val >= 0.9:
            pentgram_list.append(key)
        outputfile.write(str(key)+"\t"+str(val)+"\n")

    outputfile.close()

    outputfile = open("phrases_cleaned.txt","w")
    for l in xrange(len(tagged_sarcastic_tweets)):
        list_tweet = tokens_sarcastic_tweets[l]
        outputfile.write(str(cleaned_tweets_sarcasm[l]) + "\n")
        for i in xrange(len(bigram_list)):
            b_match = list()
            b_match = bigram_list[i].split()
            j = 0
            k = 1
            while k < len(tagged_sarcastic_tweets[l]):
                bigram = list()
                bigram.append(tagged_sarcastic_tweets[l][j])
                bigram.append(tagged_sarcastic_tweets[l][k])
                if bigram == b_match:
                    #print "bigram match"
                    if k >= len(list_tweet):
                        print cleaned_tweets_sarcasm[l]
                        print list_tweet 
                        print tagged_sarcastic_tweets[l]
                    else:
                        outputfile.write(str(list_tweet[j])+" "+str(list_tweet[k])+ "\t")
                j += 1
                k += 1
        outputfile.write("\n")
        #outputfile.write(str(cleaned_tweets_sarcasm[l]) + "\t")
        for i in xrange(len(trigram_list)):
            t_match = list()
            t_match = trigram_list[i].split()
            j = 0
            k = 1
            m = 2
            while m < len(tagged_sarcastic_tweets[l]):
                trigram = list()
                trigram.append(tagged_sarcastic_tweets[l][j])
                trigram.append(tagged_sarcastic_tweets[l][k])
                trigram.append(tagged_sarcastic_tweets[l][m])
                if trigram == t_match:
                    #print " trigam match"
                    if m >= len(list_tweet):
                        print cleaned_tweets_sarcasm[l]
                        print list_tweet 
                        print tagged_sarcastic_tweets[l]
                    else:
                        outputfile.write(str(list_tweet[j])+" "+str(list_tweet[k])+ " "+str(list_tweet[m])+"\t")
                j += 1
                k += 1
                m += 1
        outputfile.write("\n")
        for i in xrange(len(quadgram_list)):
            q_match = list()
            q_match = quadgram_list[i].split()
            j = 0
            k = 1
            m = 2
            n = 3
            while n < len(tagged_sarcastic_tweets[l]):
                quadgram = list()
                quadgram.append(tagged_sarcastic_tweets[l][j])
                quadgram.append(tagged_sarcastic_tweets[l][k])
                quadgram.append(tagged_sarcastic_tweets[l][m])
                quadgram.append(tagged_sarcastic_tweets[l][n])
                if quadgram == q_match:
                    #print " trigam match"
                    if n >= len(list_tweet):
                        print cleaned_tweets_sarcasm[l]
                        print list_tweet 
                        print tagged_sarcastic_tweets[l]
                    else:
                        outputfile.write(str(list_tweet[j])+" "+str(list_tweet[k])+ " "+str(list_tweet[m])+ " "+str(list_tweet[n])+"\t")
                j += 1
                k += 1
                m += 1
                n += 1
        outputfile.write("\n")
        for i in xrange(len(pentgram_list)):
            p_match = list()
            p_match = pentgram_list[i].split()
            j = 0
            k = 1
            m = 2
            n = 3
            o = 4
            while o < len(tagged_sarcastic_tweets[l]):
                pentgram = list()
                pentgram.append(tagged_sarcastic_tweets[l][j])
                pentgram.append(tagged_sarcastic_tweets[l][k])
                pentgram.append(tagged_sarcastic_tweets[l][m])
                pentgram.append(tagged_sarcastic_tweets[l][n])
                pentgram.append(tagged_sarcastic_tweets[l][o])
                if pentgram == p_match:
                    #print " trigam match"
                    if o >= len(list_tweet):
                        print cleaned_tweets_sarcasm[l]
                        print list_tweet 
                        print tagged_sarcastic_tweets[l]
                    else:
                        outputfile.write(str(list_tweet[j])+" "+str(list_tweet[k])+ " "+str(list_tweet[m])+ " "+str(list_tweet[n])+" "+str(list_tweet[o])+"\t")
                j += 1
                k += 1
                m += 1
                n += 1
                o += 1
        outputfile.write("\n\n\n")
    
    outputfile.close()

    
    