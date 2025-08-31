using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace DeepFace.Migrations
{
    /// <inheritdoc />
    public partial class AddConsentToFaces : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<bool>(
                name: "Consent",
                table: "Faces",
                type: "bit",
                nullable: false,
                defaultValue: false);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "Consent",
                table: "Faces");
        }
    }
}
