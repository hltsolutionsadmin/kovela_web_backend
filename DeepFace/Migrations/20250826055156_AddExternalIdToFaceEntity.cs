using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace DeepFace.Migrations
{
    /// <inheritdoc />
    public partial class AddExternalIdToFaceEntity : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<string>(
                name: "ExternalId",
                table: "Faces",
                type: "nvarchar(max)",
                nullable: false,
                defaultValue: "");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "ExternalId",
                table: "Faces");
        }
    }
}
