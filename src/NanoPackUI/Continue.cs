using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace NanoPack_UI__draft_
{
    public partial class Continue : Form
    {
        public Continue(string text)
        {
            InitializeComponent();
            label1.Text = text;
        }
    }
}
