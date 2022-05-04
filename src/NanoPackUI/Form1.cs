using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Pipes;
using System.Linq;
using System.Threading;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace NanoPack_UI__draft_
{
    public partial class Form1 : Form
    {
        public enum CellSelection { E, Type1, Type2, Type3, Type4 };

        //Addresses for Senior Design Lab
        //private const string condaActivateBat = @"C:\Users\dbids\Miniconda3\Scripts\activate.bat";

        //Addresses for Justin's Computer
        //private const string condaActivateBat = @"C:\Users\justi\miniconda3\Scripts\activate.bat";

        int gridSize = 8;
        int cellSize = 45;

        DataTable table = new DataTable();

        string csv_path = "";

        PipeClient pipe;

        bool loadingImages = false;

        public Form1()
        {
            InitializeComponent();
            string homeDir = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", ".."));
            if (!File.Exists(homeDir + @"/path.txt"))
            {
                bool uploaded = false;
                while (!uploaded) {
                    PathSelector pathSelector = new PathSelector();
                    if (pathSelector.ShowDialog() == DialogResult.OK)
                    {
                        uploaded = true;
                    }
                    else if (pathSelector.ShowDialog() == DialogResult.Cancel)
                    {
                        break;
                    }
                }
            }
            string condaLocation = File.ReadAllText(homeDir + @"/path.txt");
            Run_cmd(homeDir, condaLocation);
            pipe = new PipeClient();
            visualizationBox.Image = new Bitmap(visualizationBox.Width, visualizationBox.Height);
            tabControl1.DrawItem += new DrawItemEventHandler(tabControl1_DrawItem);
            Icon icon = Icon.ExtractAssociatedIcon(homeDir + @"\NanoPackUI\Images\NanoPack_logo_small.ico");
            this.Icon = icon;
        }

        CellSelection[,] initializeChips(string inp)
        {


            CellSelection[,] chips = new CellSelection[gridSize, gridSize];

            int it = 0;
            for (int i = 0; i < gridSize; i++)
            {
                for (int j = 0; j < gridSize; j++)
                {
                    if (inp[it] == 'A')
                        chips[i, j] = CellSelection.Type1;
                    else if (inp[it] == 'B')
                        chips[i, j] = CellSelection.Type2;
                    else if (inp[it] == 'C')
                        chips[i, j] = CellSelection.Type3;
                    else if (inp[it] == 'D')
                        chips[i, j] = CellSelection.Type4;
                    else if (inp[it] == '0')
                        chips[i, j] = CellSelection.E;
                    it++;
                }
            }
            return chips;
        }

        private void tabControl1_DrawItem(Object sender, System.Windows.Forms.DrawItemEventArgs e)
        {
            Graphics g = e.Graphics;
            Brush _textBrush;

            // Get the item from the collection.
            TabPage _tabPage = tabControl1.TabPages[e.Index];

            // Get the real bounds for the tab rectangle.
            Rectangle _tabBounds = tabControl1.GetTabRect(e.Index);

            if (e.State == DrawItemState.Selected)
            {

                // Draw a different background color, and don't paint a focus rectangle.
                _textBrush = new SolidBrush(Color.Black);
                g.FillRectangle(Brushes.LightBlue, e.Bounds);
            }
            else
            {
                _textBrush = new System.Drawing.SolidBrush(e.ForeColor);
                e.DrawBackground();
            }

            // Use our own font.
            Font _tabFont = new Font("Arial", 10.0f, FontStyle.Bold, GraphicsUnit.Pixel);

            // Draw string. Center the text.
            StringFormat _stringFlags = new StringFormat();
            _stringFlags.Alignment = StringAlignment.Center;
            _stringFlags.LineAlignment = StringAlignment.Center;
            g.DrawString(_tabPage.Text, _tabFont, _textBrush, _tabBounds, new StringFormat(_stringFlags));
        }

        private void Upload_Click(object sender, EventArgs e)
        {
            OpenFileDialog dialog = new OpenFileDialog();
            dialog.Filter = "Text files | *.csv"; // file types, that will be allowed to upload
            dialog.Multiselect = false; // allow/deny user to upload more than one file at a time
            if (dialog.ShowDialog() == DialogResult.OK) // if user clicked OK
            {
                table = new DataTable();
                csv_path = dialog.FileName;
                String name = dialog.SafeFileName;
                label1.Text = name;
                using (StreamReader reader = new StreamReader(new FileStream(csv_path, FileMode.Open), new UTF8Encoding()))
                {
                    string[] traveler_num = reader.ReadLine().Split(',');
                    string[] headers = reader.ReadLine().Split(',');
                    foreach (string header in headers)
                    {
                        table.Columns.Add(header);
                    }

                    while (reader.Peek() >= 0)
                    {
                        string[] rows = Regex.Split(reader.ReadLine(), ",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)");
                        DataRow row = table.NewRow();
                        for (int i = 0; i < headers.Length; i++)
                        {
                            row[i] = rows[i];
                        }
                        table.Rows.Add(row);
                    }
                }
            }
            dataGridView1.DataSource = table;
            label2.Text = "";
            Load_visualization();
        }

        private void dataGridView1_CellContentClick(object sender, DataGridViewCellEventArgs e)
        { }

        private void Load_visualization()
        {
            using (var g = Graphics.FromImage(visualizationBox.Image))
            {
                g.Clear(Color.White);
                StringBuilder sb = new StringBuilder("00000000000000000000000000000000000000000000000000000000000000000");
                foreach (DataRow row in table.Rows)
                {
                    if (Int32.TryParse(row[0].ToString(), out int pos))
                    {
                        //int pos = Int32.Parse(row[0].ToString());
                        char type = row[2].ToString()[0];
                        sb[pos] = type;
                    }
                }

                string inp = sb.ToString();
                Pen p = new Pen(Color.Black);
                Brush b1 = new SolidBrush(Color.LightBlue);
                Brush b2 = new SolidBrush(Color.LightGreen);
                Brush b3 = new SolidBrush(Color.LightSteelBlue);
                Brush b4 = new SolidBrush(Color.PaleGoldenrod);

                CellSelection[,] chips = initializeChips(inp);


                for (int x = 0; x <= gridSize; x++)
                {
                    g.DrawLine(p, x * cellSize, 0, x * cellSize, gridSize * cellSize);
                }
                for (int y = 0; y <= gridSize; y++)
                {
                    g.DrawLine(p, 0, y * cellSize, gridSize * cellSize, y * cellSize);
                }

                int cellOffset = 9;

                for (int i = 0; i < gridSize; i++)
                {
                    for (int j = 0; j < gridSize; j++)
                    {
                        if (chips[i, j] == CellSelection.Type1)
                        {
                            g.FillRectangle(b1, i * cellSize + cellOffset, j * cellSize + cellOffset, 30, 30);
                        }
                        else if (chips[i, j] == CellSelection.Type2)
                        {
                            g.FillRectangle(b2, i * cellSize + cellOffset, j * cellSize + cellOffset, 30, 30);
                        }
                        else if (chips[i, j] == CellSelection.Type3)
                        {
                            g.FillRectangle(b3, i * cellSize + cellOffset, j * cellSize + cellOffset, 30, 30);
                        }
                        else if (chips[i, j] == CellSelection.Type4)
                        {
                            g.FillRectangle(b4, i * cellSize + cellOffset, j * cellSize + cellOffset, 30, 30);
                        }
                    }
                }
                visualizationBox.Refresh();
            }
        }

        public void setDataSource()
        {
            listBox1.DataSource = pipe.received_messages;
            ((CurrencyManager)listBox1.BindingContext[listBox1.DataSource]).Refresh();
        }

        public static void Run_cmd(string home, string condaActivateBat)
        {
            string result = string.Empty;
            condaActivateBat = condaActivateBat.Replace(System.Environment.NewLine, string.Empty);
            try
            {
                Process process = new Process();
                process.StartInfo.FileName = "cmd.exe";
                process.StartInfo.CreateNoWindow = false;
                process.StartInfo.RedirectStandardInput = true;
                process.StartInfo.RedirectStandardOutput = false;
                process.StartInfo.UseShellExecute = false;
                process.Start();
                process.StandardInput.WriteLine(condaActivateBat);
                process.StandardInput.WriteLine("conda info --envs");
                process.StandardInput.WriteLine("activate NanoPackEnv");
                //process.StandardInput.WriteLine("cd " + home + @"/NanoPackUI");
                //process.StandardInput.WriteLine("python Control_.NET_Only.py");
                process.StandardInput.WriteLine("cd " + home);
                process.StandardInput.WriteLine("python Control.py");
                process.StandardInput.Flush();
                process.StandardInput.Close();
            }
            catch (Exception ex)
            {
                throw new Exception("R Script failed: " + result, ex);
            }
        }

        private async void Start_Packing_Click(object sender, EventArgs e)
        {
            if (csv_path == "")
            {
                MessageBox.Show("Please upload a CSV file before initiating packing", "No CSV file chosen",
                MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            else
            {

                bool isOpen = pipe.IsOpen();
                if (isOpen)
                {
                    pipe.StartDataTransfer(csv_path);
                    while (pipe.active == true)
                    {
                        setDataSource();
                        await Task.Delay(500);
                    }
                    setDataSource();
                    string text = (string)pipe.received_messages[pipe.received_messages.Count - 1];
                    if (text == "Exited with code: NormalExit\n")
                    {
                        MessageBox.Show(text, "Packing successful!",
                        MessageBoxButtons.OK, MessageBoxIcon.Information);
                    }
                    else if (text == "Exited with code: NormalExit\n" || text == "Exited with code: Other Error\n" || text == "Exited with code: KeyboardInterrupt\n" ||
                       text == "Exited with code: TimeoutError\n" || text == "Exited with code: CSVError\n" || text == "Exited with code: TooFewClamshells\n" ||
                        text == "Exited with code: TravelerNotFound\n" || text == "Exited with code: ClamshellsNotFound\n" || text == "Exited with code: TinygThreadException\n")
                    {
                        MessageBox.Show(text, "Packing failed!",
                        MessageBoxButtons.OK, MessageBoxIcon.Error);
                    }
                    else if (text[0] == 'U')
                    {
                        Continue cont = new Continue(text);
                        if (cont.ShowDialog(this) == DialogResult.Retry)
                        {
                            pipe.shouldContinue = "Continue";
                        }
                        else if (cont.ShowDialog(this) == DialogResult.Cancel)
                        {
                            pipe.shouldContinue = "Exit";
                        }
                        cont.Dispose();
                        string messageBoxText = "You won!";
                        MessageBox.Show(messageBoxText);
                    }
                }
            }
        }

        private void Cancel_Click(object sender, EventArgs e)
        {
            if (pipe.active != true)
            {
                MessageBox.Show("No jobs to cancel!", "Error",
                MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            else
            {
                pipe.Cancel();
                if (pipe.active != true)
                {
                    MessageBox.Show("Packing halted", "Success",
                    MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
            }
        }

        private void loadButton_Click(object sender, EventArgs e)
        {


            if (loadingImages == false)
            {
                loadingImages = true;
                Thread T1 = new Thread(new ThreadStart(loadImages));
                T1.Start();
                loadButton.Text = "Stop loading chip images";
            }
            else if (loadingImages == true)
            {
                loadingImages = false;
                loadButton.Text = "Load Images from Machine";
            }
        }

        private void loadImages()
        {
            while (loadingImages == true)
            {
                try {
                    //MyImage = new Bitmap(@"C:\Users\justi\Documents\GitHub\NanoView_G33\dev\machine_learning\numberRecognition\temp\chips.jpg");
                    pictureBox1.SizeMode = PictureBoxSizeMode.StretchImage;
                    // FileStream stream = new FileStream(@"C:\Users\justi\Documents\GitHub\NanoView_G33\dev\machine_learning\numberRecognition\temp\chips.jpg", FileMode.Open, FileAccess.Read);
                    FileStream stream = new FileStream(@"C:\NanoView_G33\src\output.png", FileMode.Open, FileAccess.Read);
                    pictureBox1.Image = Image.FromStream(stream);
                    stream.Close();
                    //pictureBox1.Image = (Image) MyImage;
                    Thread.Sleep(1000);
                }
                catch(Exception)
                {
                    Console.WriteLine("Image can't load, was being written to");
                    Thread.Sleep(1000);
                }
            }
        }

    }
}