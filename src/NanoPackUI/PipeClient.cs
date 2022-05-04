using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.IO.Pipes;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NanoPack_UI__draft_
{
    class PipeClient
    {
        public string retrieved_info = "";
        public ArrayList received_messages = new ArrayList();
        public NamedPipeClientStream pipeClient;
        public bool active = false;
        private bool cancelled = false;
        public string shouldContinue = "unknown";
        public PipeClient()
        {
            pipeClient = new NamedPipeClientStream(".", "Test Pype1", PipeDirection.InOut);


            // Connect to the pipe or wait until the pipe is available.
            Console.Write("Attempting to connect to pipe...");
            if (pipeClient.IsConnected != true)
            {
                pipeClient.Connect();
            }
            if (pipeClient.IsConnected != true)
            {
                Console.WriteLine("This is fucked");
            }


            //StartDataTransfer(@"C:\Users\justi\Documents\GitHub\NanoView_G33\dev\first_prototype\Control.py");


        }

        public async void StartDataTransfer(string csv_path)
        {
            try
            {
                StreamReader _sr = new StreamReader(pipeClient);
                StreamWriter _sw = new StreamWriter(pipeClient);

                if (pipeClient.IsConnected != true)
                {
                    Console.WriteLine("This is fucked");
                }
                // Display the read text to the console
                string temp = "";

                active = true;

                char[] buf = new char[1048576];
                cancelled = false;
                while (!cancelled)
                {
                    int num_received_bytes = await _sr.ReadAsync(buf, 0, 1048576);
                    if (num_received_bytes > 0)
                    {
                        temp = new string(buf, 0, num_received_bytes);
                    }
                    if (temp == "Exited with code: NormalExit\n" || temp == "Exited with code: Other Error\n" || temp == "Exited with code: KeyboardInterrupt\n" ||
                       temp == "Exited with code: TimeoutError\n" || temp == "Exited with code: CSVError\n" || temp == "Exited with code: TooFewClamshells\n" ||
                        temp == "Exited with code: TravelerNotFound\n" || temp == "Exited with code: ClamshellsNotFound\n" || temp == "Exited with code: TinygThreadException\n")
                    {
                        received_messages.Add(temp);
                        break;
                    }
                    if (temp != "")
                        received_messages.Add(temp);
                    else break;
                    if (temp == "CSVRequested\n")
                    {
                        _sw.WriteLine(csv_path);
                        _sw.Flush();
                    }
                    else if (temp[0] == 'S')
                    {
                        _sw.WriteLine("Data Received");
                        _sw.Flush();
                    }
                    else if (temp[0] == 'U')
                    {
                        while (shouldContinue == "unknown")
                        {
                            await Task.Delay(500);
                        }
                        _sw.WriteLine(shouldContinue);
                        _sw.Flush();
                    }
                    else
                    {
                        _sw.WriteLine("Haven't specifically handled yet");
                        _sw.Flush();
                    }
                }
                if (cancelled)
                {
                    _sw.WriteLine("STOP");
                    _sw.Flush();
                }
                active = false;

            }
            catch (Exception ex)
            {
                throw new Exception("No data received", ex);
            }
        }

        public void Cancel()
        {
            if (pipeClient.IsConnected != true)
            {
                Console.WriteLine("This is fucked");
            }
            else
            {
                cancelled = true;
            }
        }

        public bool IsOpen()
        {
            return pipeClient.IsConnected;
        }

        public void ClosePipe()
        {
            pipeClient.Close();
        }

    }
}