import argparse
import configparser


# def verify(config_file_args, config_file_name):


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--configfile', type=str, default='./config_files_mesocenter/percentN_randomly/percent1/simple/SSL_simple_percent1_randomly_02.cfg')
    args = parser.parse_args()

    args.config_file_name = args.configfile.rsplit("/", 1)[1]

    if args.configfile:
        config = configparser.RawConfigParser()
    else:
        raise ValueError(f"{args.configFile} does not exist.")
    print(f"\n\n*************************************{args.config_file_name}****************************************")
    print(config.read(f"{args.configfile}"))
    # verify(config_file_args=config_file_arguments, config_file_name=args.config_file_name)
    # print("\nConfiguration File Arguments:\n", json.dumps(config, indent=2, separators=(",", ":")))
