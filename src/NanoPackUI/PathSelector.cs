using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace NanoPack_UI__draft_
{
    public partial class PathSelector : Form
    {
        public PathSelector()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            string homeDir = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", ".."));
            OpenFileDialog dialog = new OpenFileDialog();
            dialog.Filter = "activate.bat | *.bat"; // file types, that will be allowed to upload
            dialog.Multiselect = false; // allow/deny user to upload more than one file at a time
            if (dialog.ShowDialog() == DialogResult.OK) // if user clicked OK
            {
                String bat = dialog.FileName;
                String name = dialog.SafeFileName;
                label2.Text = name;
                if (label2.Text == "activate.bat")
                {
                    button2.Enabled = true;
                    label3.Text = "";
                }
                else
                {
                    label3.Text = "Incorrect file";
                }
                using (StreamWriter outputFile = new StreamWriter(Path.Combine(homeDir, "path.txt")))
                {
                    outputFile.WriteLine(bat);
                }
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            if (label2.Text != "activate.bat")
            {
                label3.Text = "Please select the activate.bat for your Anaconda installation";
                button2.DialogResult = DialogResult.Cancel;
            }
            else
            {
                label3.Text = "";
                button2.DialogResult = DialogResult.OK;
            }
        }
    }
}
