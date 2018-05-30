# /usr/bin/env python
"""
Simple synthetic data generator
author: John Ding johnding1996@hotmail.com

The purpose of this script is to attempt to create synthetic JSON tracking logs
that are equally good to test through the apipe and qpipe of MLC to MLQ and MLM.

There is a distribution added for dropout.

"""
import sys
import json
import os
import string
import random
import argparse
from datetime import datetime, timedelta

COURSE_NAME_DEFAULT = 'synthetic'
OUT_PATH_DEFAULT = os.path.join(os.path.realpath(__file__), '../../..')
NUM_RECORDS_DEFAULT = 10000
SHALL_GEN_VISMOOC_DEFAULT = False
SHALL_GEN_NEWMITX_DEFAULT = False

DEFAULT_DATA_FILE_SUFFIX = {'log_file': '_log_data.json',
                            'vismooc_file': {
                                'course_structure_file': '-course_structure-prod-analytics.json',
                                'course_certificates_file': '-certificates_generatedcertificate-prod-analytics.sql',
                                'course_enrollment_file': '-student_courseenrollment-prod-analytics.sql',
                                'course_user_file': '-auth_user-prod-analytics.sql',
                                'course_profile_file': '-auth_userprofile-prod-analytics.sql',
                                'course_forum_file': '-prod.mongo',
                            },
                            'newmitx_file': {
                            },
                            }

AGENT_TYPES = [
    "Mozilla\/5.0 (Windows NT 6.1; WOW64) AppleWebKit\/537.36 (KHTML, like Gecko) Chrome\/39.0.2171.95 Safari\/537.36",
    "Dalvik/1.6.0 (Linux; U; Android 4.0.2; sdk Build/ICS_MR0)",
    "Mozilla\/5.0 (Windows NT 6.1; WOW64; Trident\/7.0; rv:11.0) like Gecko",
    "Mozilla\/5.0 (X11; Ubuntu; Linux x86_64; rv:37.0) Gecko\/20100101 Firefox\/37.0",
]
N_USERS = 1000
# TODO yield this
N_TIMES = 10000


def gen_log(num_records, out_path, course_name):
    out_filename = os.path.join(out_path, course_name + DEFAULT_DATA_FILE_SUFFIX['log_file'])
    with os.fdopen(
            os.open(out_filename, os.O_CREAT | os.O_TRUNC | os.O_WRONLY),
            'w') as outfile:
        course_code = str(random.randint(1, 20)) + u"." + str(random.randint(0, 999)) + u'x'
        course_term = str(random.randint(2000, 2016)) + u'_' + random.choice([u"Fall", u"Spring"])
        usernames = []
        for username in range(N_USERS):
            usernames.append(u''.join(random.choice(string.ascii_letters + string.digits) for _ in range(random.randint(0, 10))))

        times = []
        time_format = u"%Y-%m-%d %H:%M:%S.%f %Z"
        start_time = u"{0}-{1}-{2} {3}:{4}:{5}.{6} UTC".format(
                str(2017),
                str(1).zfill(2),
                str(1).zfill(2),
                str(0).zfill(2),
                str(0).zfill(2),
                str(0).zfill(2),
                str(0).zfill(6)
            )
        start_time = datetime.strptime(start_time, time_format)
        # Duration is nine weeks
        duration = 9.0 * 7.0
        # Mean dropout (currently mean activity) 3rd week
        _lambda = 3.0 / duration        
        for _ in range(N_TIMES):
            _time = timedelta(days=random.expovariate(_lambda),
                                       seconds=random.randint(0, 59),
                                       hours=random.randint(0, 23),
                                       minutes=random.randint(0, 59)
                                       )
            _time = start_time + _time
            _time = "{}UTC".format(datetime.strftime(_time, time_format))
            times.append(unicode(_time))

        for count in range(num_records):
            encoded_json_line = "%s\n" % gen_one_resource_log(course_code, course_term, usernames, times)
            encoded_json_line += "%s\n" % gen_one_click_log(course_code, course_term, usernames, times)
            encoded_json_line += "%s\n" % gen_one_submission_log(course_code, course_term, usernames, times)
            encoded_json_line += "%s\n" % gen_one_assessment_log(course_code, course_term, usernames, times)
            outfile.write(encoded_json_line)


def get_event_base(course_code, course_term, usernames, times):
    l = dict()
    l[u"username"] = random.choice(usernames)
    l[u'agent'] = random.choice(AGENT_TYPES)
    l[u"event_source"] = u"server"
    l[u"ip"] = u"{0}.{1}.{2}.{3}".format(str(random.randint(0, 255)),
                                         str(random.randint(0, 255)),
                                         str(random.randint(0, 255)),
                                         str(random.randint(0, 255)))
    l[u"time"] = random.choice(times)
    l[u"course_id"] = u"MITx/{0}/{1}".format(course_code,
                                             course_term)
    l[u"_id"] = {u"$oid": u''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(24))}
    return l


def gen_one_resource_log(course_code, course_term, usernames, times):
    l = get_event_base(course_code, course_term, usernames, times)
    l[u"event_type"] = u"/courses/MITx/{0}/{1}/{2}".format(course_code,
                                                           course_term,
                                                           u"".join(random.choice(string.ascii_letters + string.digits)
                                                                    for _ in range(random.randint(0, 10))))
    l[u"event"] = {u"POST": {}, u"GET": {}}

    return json.dumps(l)


def gen_one_click_log(course_code, course_term, usernames, times):
    CLICK_EVENT_TYPES = [
        'play_video', 'load_video', 'pause_video', 'stop_video',
        'seek_video', 'speed_change_video', 'hide_transcript', 'show_transcript',
        'video_hide_cc_menu', 'video_show_cc_menu'
    ]
    l = get_event_base(course_code, course_term, usernames, times)
    l[u"event_type"] = random.choice(CLICK_EVENT_TYPES)
    l[u"event"] = {
        u'id': u''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(24)),
        u'code': u''.join(random.choice(string.ascii_letters) for _ in range(12)),
        u'currentTime': str(random.randint(0, 3600))
    }
    l[u'context'] = {
        u'org_id': u'MITx',
        u'path': u'/event',
        u'course_id': u"MITx/{0}/{1}".format(course_code,
                                             course_term),
        u'user_id': str(random.randint(0, 99999999)).zfill(8)
    }

    return json.dumps(l)


def gen_one_submission_log(course_code, course_term, usernames, times):
    SUBMISSION_EVENT_TYPES = [
        'problem_check', 'problem_check_fail', 'problem_reset',
        'reset_problem', 'problem_save', 'problem_show', 'showanswer',
        'save_problem_fail', 'save_problem_success', 'problem_graded',
        'i4x_problem_input_ajax', 'i4x_problem_problem_check',
        'i4x_problem_problem_get', 'i4x_problem_problem_reset',
        'i4x_problem_problem_save', 'i4x_problem_problem_show',
        'oe_hide_question', 'oe_show_question', 'rubric_select',
        'i4x_combinedopenended_some_action', 'peer_grading_some_action',
        'staff_grading_some_action', 'i4x_peergrading_some_action'
    ]
    l = get_event_base(course_code, course_term, usernames, times)
    l[u"event_type"] = random.choice(SUBMISSION_EVENT_TYPES)
    l[u"event"] = {
        u'problem': u''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(24)),
    }
    l[u'context'] = {
        u'org_id': u'MITx',
        u'path': u'/event',
        u'course_id': u"MITx/{0}/{1}".format(course_code,
                                             course_term),
        u'user_id': str(random.randint(0, 99999999)).zfill(8)
    }

    return json.dumps(l)


def gen_one_assessment_log(course_code, course_term, usernames, times):
    ASSESSMENT_EVENT_TYPES = [
        'save_problem_check', 'problem_check'
    ]
    l = get_event_base(course_code, course_term, usernames, times)
    l[u"event_type"] = random.choice(ASSESSMENT_EVENT_TYPES)
    problem_id = u''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(24))
    l[u"event"] = {
        u'answers': {problem_id: u'choice_0'},
        u'attempts': random.randint(0, 100),
        u'correct_map': {problem_id: {
            u'correctness': random.choice([u'correct', u'incorrect']),
            u'hint': u'',
            u'hintmode': None,
            u'msg': u'',
            u'npoints': 0,
            u'queuestate': None
        }},
        u'grade': 0,
        u'max_grade': 0,
        u'problem_id': problem_id,
        u'state': {},
        u'success': random.choice([u'correct', u'incorrect']),
    }
    l[u'context'] = {
        u'org_id': u'MITx',
        u'path': u'/event',
        u'course_id': u"MITx/{0}/{1}".format(course_code,
                                             course_term),
        u'user_id': str(random.randint(0, 99999999)).zfill(8)
    }

    return json.dumps(l)


def create_directories(out_path, course_name):
    out_path = os.path.abspath(out_path)
    if not os.path.isdir(out_path):
        sys.exit('Output data directory does not exist')

    out_path = os.path.join(out_path + "/", 'data')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_path = os.path.join(out_path + "/", course_name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print('Generated synthetic course path: %s' % out_path)
    out_path = os.path.join(out_path + "/", 'log_data')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Generate sample synthetic data to test MPL components MLC, MLQ and MLM for dropout.')
    parser.add_argument("-c", action="store", default=COURSE_NAME_DEFAULT, dest='course_name',
                        help='synthethic course name')
    parser.add_argument('-o', action='store', default=OUT_PATH_DEFAULT,
                        dest='out_path', help='path to data folder, default: the parent folder of MLC')
    parser.add_argument('-n', action='store', default=NUM_RECORDS_DEFAULT, type=int, dest='num_records',
                        help='number of records of all synthetic data files, default 1000')
    parser.add_argument('-v', action='store_true', default=SHALL_GEN_VISMOOC_DEFAULT, dest='shall_gen_vismooc',
                        help='switch on to generate synthetic data files for vismooc extensions')
    parser.add_argument('-m', action='store_true', default=SHALL_GEN_NEWMITX_DEFAULT, dest='shall_gen_newmitx',
                        help='switch on to generate synthetic data files for newmitx extensions')
    course_name = parser.parse_args().course_name
    out_path = parser.parse_args().out_path
    num_records = parser.parse_args().num_records
    shall_gen_vismooc = parser.parse_args().shall_gen_vismooc
    shall_gen_newmitx = parser.parse_args().shall_gen_newmitx

    out_path = create_directories(out_path, course_name)
    
    gen_log(num_records, out_path, course_name)
