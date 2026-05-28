#!/usr/bin/env python

'''
Original author: lanquarden (Apr 4, 2012)
Modified by: Ryan Katnoria (Nov 21, 2025)

'''

import sys
import argparse
import driver
from src import config, network, telemetry, ibm_granite

def perform_handshake(sock, host_ip, host_port, bot_id, driver_inst):
    # Handles the initial connection sequence with the server.
    while True:
        print 'Sending id to server: ', bot_id
        buf = bot_id + driver_inst.init()
        print 'Sending init string to server:', buf
        
        try:
            sock.sendto(buf, (host_ip, host_port))
            buf, addr = sock.recvfrom(1000)
        except network.socket.error:
            print "Waiting for server..."
            continue
            
        if buf.find('***identified***') >= 0:
            print 'Received: ', buf
            break

def main():
    # 1. Argument parser.
    parser = argparse.ArgumentParser(description='Python client for TORCS SCRC')
    parser.add_argument('--host', action='store', dest='host_ip', default=config.DEFAULT_HOST, help='Host IP')
    parser.add_argument('--port', action='store', type=int, dest='host_port', default=config.DEFAULT_PORT, help='Host port')
    parser.add_argument('--id', action='store', dest='id', default=config.DEFAULT_ID, help='Bot ID')
    parser.add_argument('--maxEpisodes', action='store', dest='max_episodes', type=int, default=1, help='Max episodes')
    parser.add_argument('--maxSteps', action='store', dest='max_steps', type=int, default=0, help='Max steps')
    parser.add_argument('--track', action='store', dest='track', default=None, help='Track name')
    parser.add_argument('--stage', action='store', dest='stage', type=int, default=config.DEFAULT_STAGE, help='Stage')
    
    args = parser.parse_args()

    print 'Connecting to:', args.host_ip, '@ port:', args.host_port
    print 'Bot ID:', args.id
    print '*********************************************'

    # 2. Component initialiser.
    sock = network.create_socket()
    ai_client = ibm_granite.GraniteClient()
    d = driver.Driver(args.stage)
    
    shutdown_client = False
    cur_episode = 0
    verbose = False

    # 3. Main application loop.
    while not shutdown_client:
        
        # Connects to server.
        perform_handshake(sock, args.host_ip, args.host_port, args.id, d)
        
        current_step = 0
        
        # Race Loop.
        while True:
            buf = None
            try:
                buf, addr = sock.recvfrom(1000)
            except network.socket.error:
                print "didn't get response from server..."

            if verbose: print 'Received: ', buf

            # Checks for special server commands.
            if buf and buf.find('***shutdown***') >= 0:
                d.onShutDown()
                shutdown_client = True
                print 'Client Shutdown'
                break
            
            if buf and buf.find('***restart***') >= 0:
                d.onRestart()
                print 'Client Restart'
                break
            
            current_step += 1
            
            if current_step != args.max_steps:
                if buf:
                    raw_sensor_string = buf
                    
                    # Determines Action.
                    buf = d.drive(buf)
                    
                    # AI Logic (Every 50 steps).
                    if current_step % 50 == 0:
                        data_dict = telemetry.parse_raw_telemetry(raw_sensor_string)
                        metrics = {
                            "step": current_step,
                            "track": args.track,
                            "sensors": data_dict
                        }
                        ai_client.send_async_analysis(metrics)
            else:
                buf = '(meta 1)'

            if verbose: print 'Sending: ', buf
            
            if buf:
                try:
                    sock.sendto(buf, (args.host_ip, args.host_port))
                except network.socket.error:
                    print "Failed to send data...Exiting..."
                    sys.exit(-1)
        
        cur_episode += 1
        if cur_episode == args.max_episodes:
            shutdown_client = True

    sock.close()

if __name__ == '__main__':
    main()