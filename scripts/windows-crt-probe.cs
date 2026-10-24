// Test helper compiled into the staged application directory, run in a fresh
// process, and removed before packaging. It must resolve ONNX Runtime's C++
// dependencies from that directory rather than the build machine's System32.
using System;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

internal static class WindowsCrtProbe
{
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, ExactSpelling = true, SetLastError = true)]
    private static extern IntPtr LoadLibraryW(string path);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, ExactSpelling = true, SetLastError = true)]
    private static extern IntPtr GetModuleHandleW(string name);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, ExactSpelling = true, SetLastError = true)]
    private static extern uint GetModuleFileNameW(IntPtr module, StringBuilder path, int size);

    private static int Main(string[] arguments)
    {
        try
        {
            if (arguments.Length != 1)
                throw new ArgumentException("Pass the packaged onnxruntime.dll path");
            if (LoadLibraryW(Path.GetFullPath(arguments[0])) == IntPtr.Zero)
                throw new Win32Exception(Marshal.GetLastWin32Error(), "Cannot load packaged ONNX Runtime");
            foreach (string name in new[] { "msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll" })
            {
                IntPtr module = GetModuleHandleW(name);
                if (module == IntPtr.Zero)
                    throw new InvalidOperationException("ONNX Runtime dependency was not loaded: " + name);
                StringBuilder path = new StringBuilder(32768);
                if (GetModuleFileNameW(module, path, path.Capacity) == 0)
                    throw new Win32Exception(Marshal.GetLastWin32Error());
                string actual = Path.GetFullPath(path.ToString());
                string expected = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, name);
                if (!String.Equals(actual, expected, StringComparison.OrdinalIgnoreCase))
                    throw new InvalidOperationException(name + " resolved outside the package: " + actual);
                Console.WriteLine(name + "\t" + actual);
            }
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine(error.Message);
            return 1;
        }
    }
}
